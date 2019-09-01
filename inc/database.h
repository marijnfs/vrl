#ifndef __DB_H__
#define __DB_H__

#include "unqlite.h"

#include "bytes.h"

#include <string>
#include <cassert>

struct DB {
	DB(std::string path_);
	~DB();

	bool put(Bytes &id, Bytes &data);

	bool del(Bytes &id);

	bool get(Bytes &id, Bytes *data);


	unqlite *db;
	std::string path;
};

#endif
