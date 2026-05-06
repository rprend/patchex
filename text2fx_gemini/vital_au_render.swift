import AVFoundation
import AudioToolbox
import Foundation

struct Note: Codable {
    let note: UInt8
    let start: Double
    let duration: Double
    let velocity: UInt8
}

struct RenderRequest: Codable {
    let sample_rate: Double
    let duration: Double
    let notes: [Note]
    let output_path: String
}

func fourCC(_ text: String) -> OSType {
    var result: UInt32 = 0
    for scalar in text.utf8 {
        result = (result << 8) + UInt32(scalar)
    }
    return result
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: vital_au_render <request.json>")
}

let requestURL = URL(fileURLWithPath: CommandLine.arguments[1])
let request: RenderRequest
do {
    request = try JSONDecoder().decode(RenderRequest.self, from: Data(contentsOf: requestURL))
} catch {
    fail("failed to read request: \(error)")
}

let componentDescription = AudioComponentDescription(
    componentType: kAudioUnitType_MusicDevice,
    componentSubType: fourCC("Vita"),
    componentManufacturer: fourCC("Tyte"),
    componentFlags: 0,
    componentFlagsMask: 0
)

let semaphore = DispatchSemaphore(value: 0)
var loadedUnit: AVAudioUnit?
var loadError: Error?
AVAudioUnit.instantiate(with: componentDescription, options: []) { unit, error in
    loadedUnit = unit
    loadError = error
    semaphore.signal()
}
semaphore.wait()
if let loadError {
    fail("failed to instantiate Vital AU: \(loadError)")
}
guard let unit = loadedUnit else {
    fail("failed to instantiate Vital AU: no unit returned")
}
guard let midiUnit = unit as? AVAudioUnitMIDIInstrument else {
    fail("Vital AU did not instantiate as AVAudioUnitMIDIInstrument")
}

let engine = AVAudioEngine()
engine.attach(midiUnit)
let format = AVAudioFormat(standardFormatWithSampleRate: request.sample_rate, channels: 2)!
engine.connect(midiUnit, to: engine.mainMixerNode, format: format)
engine.connect(engine.mainMixerNode, to: engine.outputNode, format: format)

do {
    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 128)
    try engine.start()
} catch {
    fail("failed to start offline render engine: \(error)")
}

let totalFrames = AVAudioFramePosition(request.duration * request.sample_rate)
let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: engine.manualRenderingMaximumFrameCount)!
let outputURL = URL(fileURLWithPath: request.output_path)
var pcm = Data()

func appendUInt16LE(_ value: UInt16, to data: inout Data) {
    var little = value.littleEndian
    data.append(Data(bytes: &little, count: MemoryLayout<UInt16>.size))
}

func appendUInt32LE(_ value: UInt32, to data: inout Data) {
    var little = value.littleEndian
    data.append(Data(bytes: &little, count: MemoryLayout<UInt32>.size))
}

func appendFloat32LE(_ value: Float, to data: inout Data) {
    var bits = value.bitPattern.littleEndian
    data.append(Data(bytes: &bits, count: MemoryLayout<UInt32>.size))
}

func writeFloatWav(path: URL, pcm: Data, sampleRate: UInt32, channels: UInt16) throws {
    var data = Data()
    data.append("RIFF".data(using: .ascii)!)
    appendUInt32LE(UInt32(36 + pcm.count), to: &data)
    data.append("WAVE".data(using: .ascii)!)
    data.append("fmt ".data(using: .ascii)!)
    appendUInt32LE(16, to: &data)
    appendUInt16LE(3, to: &data)
    appendUInt16LE(channels, to: &data)
    appendUInt32LE(sampleRate, to: &data)
    appendUInt32LE(sampleRate * UInt32(channels) * 4, to: &data)
    appendUInt16LE(channels * 4, to: &data)
    appendUInt16LE(32, to: &data)
    data.append("data".data(using: .ascii)!)
    appendUInt32LE(UInt32(pcm.count), to: &data)
    data.append(pcm)
    try data.write(to: path)
}

struct Event {
    let frame: AVAudioFramePosition
    let note: UInt8
    let velocity: UInt8
    let on: Bool
}

var events: [Event] = []
for note in request.notes {
    let start = max(0, AVAudioFramePosition(note.start * request.sample_rate))
    let end = min(totalFrames, max(start + 1, AVAudioFramePosition((note.start + note.duration) * request.sample_rate)))
    events.append(Event(frame: start, note: note.note, velocity: note.velocity, on: true))
    events.append(Event(frame: end, note: note.note, velocity: 0, on: false))
}
events.sort { lhs, rhs in
    if lhs.frame == rhs.frame { return lhs.on && !rhs.on }
    return lhs.frame < rhs.frame
}

var eventIndex = 0
while engine.manualRenderingSampleTime < totalFrames {
    let currentFrame = engine.manualRenderingSampleTime
    while eventIndex < events.count && events[eventIndex].frame <= currentFrame {
        let event = events[eventIndex]
        if event.on {
            midiUnit.startNote(event.note, withVelocity: event.velocity, onChannel: 0)
        } else {
            midiUnit.stopNote(event.note, onChannel: 0)
        }
        eventIndex += 1
    }
    let framesRemaining = totalFrames - currentFrame
    let framesToRender = min(buffer.frameCapacity, AVAudioFrameCount(framesRemaining))
    buffer.frameLength = framesToRender
    do {
        let status = try engine.renderOffline(framesToRender, to: buffer)
        if status == .success {
            let frames = Int(buffer.frameLength)
            if let channels = buffer.floatChannelData, frames > 0 {
                for frame in 0..<frames {
                    appendFloat32LE(channels[0][frame], to: &pcm)
                    appendFloat32LE(channels[min(1, Int(buffer.format.channelCount) - 1)][frame], to: &pcm)
                }
            }
        } else if status == .cannotDoInCurrentContext {
            continue
        } else if status == .insufficientDataFromInputNode {
            continue
        } else {
            fail("offline render failed with status \(status)")
        }
    } catch {
        fail("offline render error: \(error)")
    }
}

engine.stop()
do {
    try writeFloatWav(path: outputURL, pcm: pcm, sampleRate: UInt32(request.sample_rate), channels: 2)
} catch {
    fail("failed to write wav: \(error)")
}
print(request.output_path)
