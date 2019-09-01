#include "database.h"

DB::DB(std::string path_) : path(path_) {
	int rc = unqlite_open(&db, "test.db", UNQLITE_OPEN_CREATE);
	assert(rc != UNQLITE_OK);
}

DB::~DB() {
	unqlite_close(db);
}

bool DB::put(Bytes &id, Bytes &data) {
	int rc = unqlite_kv_store(db, id.ptr<char*>(), id.size(), data.ptr<char*>(), data.size());
	if (rc != UNQLITE_OK)
		return false;

	return true;
}

bool DB::del(Bytes &id) {
	unqlite_kv_delete(db, id.ptr<char*>(), id.size());
}

bool DB::get(Bytes &id, Bytes *data) {
	unqlite_int64 nsize(0);
	int rc = unqlite_kv_fetch(db, id.ptr<char*>(), id.size(), NULL, &nsize);
	if (rc != UNQLITE_OK)
		return false;

	data->resize(nsize);
	rc = unqlite_kv_fetch(db, id.ptr<char*>(), id.size(), data->ptr<char*>(), &nsize);
	if (rc != UNQLITE_OK)
		return false;
}
