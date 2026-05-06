# Description Axes for Reference-Matched Synthesis

The goal is to describe a short reference section in a way that can be turned
into synth, pattern, and effects parameters. These axes are deliberately
production-oriented: each one maps to measurable audio features and to controls
we can render.

## 1. Spectral Tone

What the frequency balance feels like.

Common phrases:

- dark
- warm
- muffled
- balanced
- bright
- airy
- tinny
- boomy
- thin
- full-bodied

Useful measurements:

- low/mid/high band energy
- spectral centroid
- spectral rolloff
- spectral slope

Likely controls:

- oscillator waveform brightness
- filter cutoff
- EQ low/mid/high gains
- saturation tone

Examples:

- `bright`: more high band energy, higher filter cutoff, high shelf boost.
- `warm`: more low-mid energy, softer highs, mild saturation.
- `muffled`: reduced highs, lower filter cutoff.

## 2. Harmonic Texture

How clean, dense, noisy, or distorted the sound is.

Common phrases:

- clean
- smooth
- rounded
- analog
- warm
- crunchy
- gritty
- raspy
- distorted
- noisy
- metallic

Useful measurements:

- spectral flatness
- high-frequency harmonic energy
- zero-crossing rate
- harmonic-to-noise tendency

Likely controls:

- sine-to-square or wavetable position
- FM amount
- noise amount
- saturation drive
- distortion drive

Examples:

- `smooth`: sine/triangle-like wave, low drive.
- `crunchy`: more square-wave content, more saturation.
- `noisy`: add noise oscillator or noisy sample layer.

## 3. Envelope and Articulation

How notes start, hold, and end.

Common phrases:

- soft attack
- sharp attack
- plucky
- staccato
- sustained
- swelling
- percussive
- gated
- legato

Useful measurements:

- attack time
- transient strength
- decay slope
- RMS envelope shape
- onset density

Likely controls:

- amp attack/decay/sustain/release
- filter envelope amount
- note gate length
- compressor attack/release

Examples:

- `plucky`: fast attack, short decay, low sustain.
- `pad-like`: slower attack, high sustain, longer release.
- `gated`: short gate, sharp release.

## 4. Space and Depth

How close, wide, reverberant, or distant the sound feels.

Common phrases:

- dry
- close
- intimate
- wide
- spacious
- distant
- washed-out
- roomy
- hall-like
- atmospheric

Useful measurements:

- stereo width
- reverb tail energy
- direct-to-reverberant ratio
- decay time estimate

Likely controls:

- reverb room size
- reverb wet level
- reverb damping
- delay mix
- delay feedback
- stereo spread

Examples:

- `close`: low reverb wet level, low delay.
- `spacious`: moderate reverb and/or delay, wider stereo.
- `washed-out`: high wet level, longer room/tail.

## 5. Motion and Modulation

How much the sound changes over time.

Common phrases:

- static
- pulsing
- wobbling
- shimmering
- evolving
- phased
- chorused
- tremolo
- sidechained

Useful measurements:

- spectral flux
- RMS modulation rate
- periodic amplitude movement
- periodic brightness movement

Likely controls:

- LFO rate/depth
- filter modulation
- tremolo
- chorus/phaser
- sidechain compression
- automation curves

Examples:

- `pulsing`: rhythmic amplitude or filter modulation.
- `evolving`: slow LFO on filter/wavetable position.
- `sidechained`: periodic gain ducking against the beat.

## 6. Rhythm and Pattern

What the note pattern is doing.

Common phrases:

- straight
- syncopated
- driving
- offbeat
- arpeggiated
- repetitive
- sparse
- dense
- bouncy
- motorik

Useful measurements:

- onset density
- inter-onset intervals
- tempo
- syncopation
- pitch contour
- note duration

Likely controls:

- MIDI note sequence
- grid resolution
- gate length
- velocity pattern
- rests
- octave jumps
- swing

Examples:

- `arpeggiated`: repeated broken chord tones.
- `driving`: dense grid, consistent velocity, short rests.
- `sparse`: fewer onsets, longer gaps.

## 7. Mix Role

What job the sound has in the arrangement.

Common phrases:

- foreground
- background
- supportive
- lead-like
- bass foundation
- pad bed
- percussive layer
- hook
- texture

Useful measurements:

- loudness
- frequency occupancy
- transient dominance
- stereo width
- masking profile

Likely controls:

- gain
- EQ placement
- compression
- stereo width
- reverb send
- pattern density

Examples:

- `foreground hook`: clear mids/highs, defined envelope, controlled space.
- `background pad`: softer attack, wider space, less transient energy.
- `bass foundation`: low-frequency focus, mono/centered, limited reverb.

## Recommended Target JSON

An analysis pass should produce this shape:

```json
{
  "spectral_tone": ["warm", "slightly bright"],
  "harmonic_texture": ["smooth", "mildly saturated"],
  "envelope": ["soft attack", "medium decay"],
  "space": ["moderately wide", "short room"],
  "motion": ["pulsing", "subtle filter movement"],
  "rhythm": ["repeating arpeggio", "16th-note pulse"],
  "mix_role": ["foreground synth hook"]
}
```

The words are not just labels. They are constraints the recipe generator and
scorer use to choose synth, pattern, and effects settings.
