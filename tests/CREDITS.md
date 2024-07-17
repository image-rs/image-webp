This document describes how the images in the tests directory were obtained.

# Source images

The files in the _images_ directory are WebP images used to test the decoder.

## images/gallery1

Downloaded from https://developers.google.com/speed/webp/gallery1.

## images/gallery2

Downloaded from https://developers.google.com/speed/webp/gallery2

## images/animated

Manually created using imagemagick...

random.webp:
```
convert -delay 10 -size 64x63 xc: xc: xc: +noise Random -define webp:lossless=true random_lossless.webp
```

random2.webp:
```
convert -delay 15 -size 99x87 xc: xc: xc: xc: +noise Random -define webp:lossless=false random_lossy.webp
```

## images/regression

color_index.webp: Manually constructed to reproduce decoding error.
tiny.webp: Provided in a [bug report](https://github.com/image-rs/image-webp/issues/81).

# Reference images

These files are all PNGs with contents that should exactly match the associated WebP file in the _images_ directory.

## reference/gallery1 and reference/gallery2

These files were all produced by running dwebp with the `-nofancy` option.

## reference/animated

random-lossless-N.png:

```
for i in {1..3}; do webpmux -get frame ${i} ../../images/animated/random_lossless.webp -o random_lossless-${i}.png && convert random_lossless-${i}.png random_lossless-${i}.png; done
```

random-lossy-N.png:

```
for i in {1..4}; do webpmux -get frame ${i} ../../images/animated/random_lossy.webp -o random_lossy-${i}.png && dwebp random_lossy-${i}.png -nofancy -o random_lossy-${i}.png; done
```

## reference/regression

color_index.png: Converted with dwebp.
tiny.png: Converted with dwebp.