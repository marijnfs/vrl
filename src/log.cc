#include "log.h"

using namespace std;

Log::Log(string filename) : file(filename)
{}

template <typename T1, typename T2, typename T3>
void Log::print(T1 a, T2 b, T3 c)
{
	file << a << b << c;
}

bool exists(std::string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}


string date_string()
{
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,80,"%Y-%m-%d-%H:%M",timeinfo);
  string str(buffer);
  cout << str << std::endl;

  return str;
}
