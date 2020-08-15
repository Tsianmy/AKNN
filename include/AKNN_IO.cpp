#include "AKNN_IO.h"
#include <cstring>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/mman.h>
#include <fcntl.h>
#else
#include <windows.h>
#define stat _stat64
#define fstat _fstat64
#define fileno _fileno
#endif

namespace aknn {

void AKNN_IO::fvecs_read(const char *fname, float * & x, idx_t & n, idx_t & d)
{
	FILE *f = fopen(fname, "rb");
	if (!f) {
		perror(fname);
		exit(-1);
	}
	fread(&d, 4, 1, f);
	assert_aknn((d > 0 && d < 1000000), "unreasonable dimension");

	struct stat st;
	fstat(fileno(f), &st);
	size_t sz = st.st_size;
	assert_aknn(sz % ((d + 1) * 4) == 0, "weird file size");
	n = sz / ((d + 1) * 4);

	fseek(f, 0, SEEK_SET);
	x = new float[n * d];
	for (size_t i = 0; i < n; i++) {
		fseek(f, 4, SEEK_CUR);
		fread(x + i * d, 4, d, f);
	}

	fclose(f);
}

void AKNN_IO::ivecs_read(const char * fname, idx_t * & x, idx_t & n, idx_t & d)
{
	fvecs_read(fname, (float * &)x, n, d);
}

void AKNN_IO::codes_read(const char * fname, uint8_t * & x, idx_t & n, idx_t & d)
{
	FILE *f = fopen(fname, "rb");
	if (!f) {
		perror(fname);
		exit(-1);
	}
	fread(&d, 4, 1, f);
	assert_aknn((d > 0 && d < 1000000), "unreasonable dimension");

	struct stat st;
	fstat(fileno(f), &st);
	size_t sz = st.st_size;
	assert_aknn((sz - 4) % (d * sizeof(uint8_t)) == 0, "weird file size");
	n = (sz - 4) / (d * sizeof(uint8_t));

	fseek(f, 4, SEEK_SET);
	x = new uint8_t[n * d];
	for (size_t i = 0; i < n; i++) {
		fread(x + i * d, sizeof(*x), d, f);
	}

	fclose(f);
}

void AKNN_IO::codes_read(const char * fname, uint16_t * & x, idx_t & n, idx_t & d)
{
	FILE *f = fopen(fname, "rb");
	if (!f) {
		perror(fname);
		exit(-1);
	}
	fread(&d, 4, 1, f);
	assert_aknn((d > 0 && d < 1000000), "unreasonable dimension");

	struct stat st;
	fstat(fileno(f), &st);
	size_t sz = st.st_size;
	assert_aknn((sz - 4) % (d * sizeof(uint16_t)) == 0, "weird file size");
	n = (sz - 4) / (d * sizeof(uint16_t));

	fseek(f, 4, SEEK_SET);
	x = new uint16_t[n * d];
	for (size_t i = 0; i < n; i++) {
		fread(x + i * d, sizeof(*x), d, f);
	}

	fclose(f);
}

void AKNN_IO::fvecs_write(const char * fname, float * x, idx_t n, idx_t d)
{
	FILE *f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		return;
	}
	for (size_t i = 0; i < n; i++) {
		fwrite(&d, 4, 1, f);
		fwrite(x + i * d, sizeof(*x), d, f);
	}

	fclose(f);
}

void AKNN_IO::ivecs_write(const char * fname, idx_t * x, idx_t n, idx_t d)
{
	FILE *f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		return;
	}
	for (size_t i = 0; i < n; i++) {
		fwrite(&d, 4, 1, f);
		fwrite(x + i * d, sizeof(*x), d, f);
	}

	fclose(f);
}

void AKNN_IO::codes_write(const char * fname, uint8_t * x, idx_t n, idx_t d)
{
	FILE *f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		return;
	}
	fwrite(&d, 4, 1, f);
	fwrite(x, sizeof(*x), n * d, f);

	fclose(f);
}

void AKNN_IO::codes_write(const char * fname, uint16_t * x, idx_t n, idx_t d)
{
	FILE *f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		return;
	}
	fwrite(&d, 4, 1, f);
	fwrite(x, sizeof(*x), n * d, f);

	fclose(f);
}

void AKNN_IO::fvecs_mmap(const char * fname, float *& x, idx_t & n, idx_t & d)
{
#ifdef __linux__
	int fd = open(fname, O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "open file error\n");
		exit(-1);
	}
	struct stat statbuf;
	stat(fname, &statbuf);
	size_t size = statbuf.st_size;
	x = (float *)mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
	memcpy(&d, x, 4);
	n = (size - 4) / (4 * d);
	x++;
#else
	HANDLE dumpFileDescriptor = CreateFileA(fname, GENERIC_READ | GENERIC_WRITE,
		FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	DWORD size = GetFileSize(dumpFileDescriptor, NULL);
	HANDLE fileMappingObject = CreateFileMapping(dumpFileDescriptor, NULL, PAGE_READWRITE,
		0, 0, NULL);
	x = (float *)MapViewOfFile(fileMappingObject, FILE_MAP_READ, 0, 0, 0);
	memcpy(&d, x, 4);
	n = (size - 4) / (4 * d);
	x++;
#endif
}

}
