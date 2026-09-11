#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "agc.h"

#pragma pack(push, 1)
typedef struct {
    char     riff_id[4];   /* "RIFF" */
    uint32_t riff_size;
    char     wave_id[4];   /* "WAVE" */
} WavRiffHeader;

typedef struct {
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} WavFmtChunk;
#pragma pack(pop)

/* Scan RIFF/WAVE chunks to find "fmt " and "data". Not just a fixed
 * 44-byte skip, so it still works if the file has extra chunks
 * (LIST/fact/etc.) before the data chunk. */
static int read_wav_header(FILE *f, WavFmtChunk *fmt, uint32_t *data_size) {
    WavRiffHeader riff;
    char chunk_id[4];
    uint32_t chunk_size;
    int got_fmt = 0, got_data = 0;

    if (fread(&riff, sizeof(riff), 1, f) != 1) return -1;
    if (memcmp(riff.riff_id, "RIFF", 4) != 0 || memcmp(riff.wave_id, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a valid WAV file\n");
        return -1;
    }

    while (fread(chunk_id, 1, 4, f) == 4) {
        if (fread(&chunk_size, sizeof(chunk_size), 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            size_t to_read = sizeof(*fmt) < chunk_size ? sizeof(*fmt) : chunk_size;
            if (fread(fmt, 1, to_read, f) != to_read) return -1;
            if (chunk_size > to_read) fseek(f, (long)(chunk_size - to_read), SEEK_CUR);
            got_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            *data_size = chunk_size;
            got_data = 1;
            break; /* PCM samples start right after this */
        } else {
            fseek(f, (long)chunk_size, SEEK_CUR);
        }

        if (chunk_size & 1) fseek(f, 1, SEEK_CUR); /* chunks are word-aligned */
    }

    if (!got_fmt || !got_data) {
        fprintf(stderr, "WAV file missing fmt or data chunk\n");
        return -1;
    }
    return 0;
}

static void write_wav_header(FILE *f, uint32_t sample_rate, uint16_t num_channels,
                              uint16_t bits_per_sample, uint32_t data_size) {
    uint32_t riff_size    = 36 + data_size;
    uint32_t byte_rate    = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align  = num_channels * bits_per_sample / 8;
    uint16_t audio_format = 1; /* PCM */
    uint32_t fmt_size     = 16;

    fwrite("RIFF", 1, 4, f);
    fwrite(&riff_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
}

int main(int argc, char **argv) {
    const char *in_path, *out_path;
    FILE *fin, *fout;
    WavFmtChunk fmt;
    uint32_t data_size, total_samples, num_frames, out_data_size;
    short in_frame[FRAME_SIZE_AGC];
    short out_frame[FRAME_SIZE_AGC];
    uint32_t nf;
    AgcState *st;
    clock_t begin, end;
    double time_spent;
    float rtf;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.wav> <output.wav>\n", argv[0]);
        return 1;
    }
    in_path  = argv[1];
    out_path = argv[2];

    fin = fopen(in_path, "rb");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }

    if (read_wav_header(fin, &fmt, &data_size) != 0) {
        fclose(fin);
        return 1;
    }

    if (fmt.audio_format != 1 || fmt.bits_per_sample != 16 || fmt.num_channels != 1) {
        fprintf(stderr, "Input must be mono 16-bit PCM WAV (got format=%u, channels=%u, bits=%u)\n",
                fmt.audio_format, fmt.num_channels, fmt.bits_per_sample);
        fclose(fin);
        return 1;
    }
    if (fmt.sample_rate != SAMPLE_RATE_AGC) {
        fprintf(stderr, "Input sample rate must be %d Hz (got %u Hz)\n",
                SAMPLE_RATE_AGC, fmt.sample_rate);
        fclose(fin);
        return 1;
    }

    total_samples = data_size / sizeof(short);
    num_frames    = total_samples / FRAME_SIZE_AGC;   /* drop trailing partial frame */
    out_data_size = num_frames * FRAME_SIZE_AGC * (uint32_t)sizeof(short);

    if (num_frames == 0) {
        fprintf(stderr, "Input file shorter than one frame (%d samples)\n", FRAME_SIZE_AGC);
        fclose(fin);
        return 1;
    }

    fout = fopen(out_path, "wb");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }
    write_wav_header(fout, SAMPLE_RATE_AGC, 1, 16, out_data_size);

    st = agc_create();

    begin = clock();
    for (nf = 0; nf < num_frames; nf++) {
        if (fread(in_frame, sizeof(short), FRAME_SIZE_AGC, fin) != (size_t)FRAME_SIZE_AGC)
            break;

        agc_process_frame(st, out_frame, in_frame);

        fwrite(out_frame, sizeof(short), FRAME_SIZE_AGC, fout);
    }
    end = clock();

    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    rtf = (float)(time_spent * SAMPLE_RATE_AGC / st->frame_cnt / FRAME_SIZE_AGC);

    printf("Processed %d frames\n", st->frame_cnt);
    printf("Total time: %.3f, rtf= %.3f\n", time_spent, rtf);
    printf("Output written to %s\n", out_path);

    fclose(fin);
    fclose(fout);
    agc_destroy(st);
    return 0;
}
