#include <iostream>
#include "bytes.h"

using namespace std;

static char hexconvtab[] = "0123456789abcdef";

ostream &operator<<(ostream &out, Bytes const &bytes) {
	string hex(bytes.size() * 2 + 1, ' ');

	size_t i, j;
	int b = 0;

	for (i = j = 0; i < bytes.size(); i++) {
		b = bytes[i] >> 4;
		hex[j++] = (char)(87 + b + (((b - 10) >> 31) & -39));
		b = bytes[i] & 0xf;
		hex[j++] = (char)(87 + b + (((b - 10) >> 31) & -39));
	}
	hex[j] = '\0';

	return out << hex;
}
