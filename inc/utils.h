#pragma once

#include <string>
#include <exception>
#include <time.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <chrono>

template <typename T>
inline T &last(std::vector<T> &vec) {
  return vec[vec.size() - 1];
}

template <typename T>
inline std::ostream &operator<<(std::ostream &out, std::vector<T> &v) {
  if (v.size() == 0) return out << "[]";
  
  out << "[" << v[0];
  for (int i(1); i < v.size(); ++i)
    out << ", " << v[i];
  return out << "]";
}

inline uint64_t now_ms() {
    std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    uint64_t ms_uint = ms.count();
    return ms_uint;
}

struct StringException : public std::exception {
	std::string content;

	StringException(std::string const &content_) : content(content_) {}
	const char* what() const throw() { return content.c_str(); };
};

struct Timer {
    Timer() {start();}
    void start() {t = clock();}    
    void reset() {t = clock();}
    double since() {return double(clock() - t) / double(CLOCKS_PER_SEC);}

    clock_t t;
};
