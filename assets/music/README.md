# Background music

Each episode plays one track from this folder, quietly under Quackers's voice, and the next
episode takes the next track (in name order). With no tracks here, episodes have the voice only.

Tracks: `.mp3`, `.m4a`, `.aac`, `.wav`, `.ogg` or `.flac`. Upbeat instrumentals without vocals work best;
a track shorter than the episode loops.

Use tracks from the YouTube Audio Library (YouTube Studio, Audio Library), filtered to
"Attribution not required". Its terms don't allow sharing the files publicly, so keep this
repository private while they are in it.

Settings: `SHORTS_MUSIC=off` turns the music off, `SHORTS_MUSIC_VOLUME` sets how loud it is
(default `0.15`, from 0 to 1), and `SHORTS_MUSIC_DIR` reads tracks from another folder.
