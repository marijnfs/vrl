#ifndef __WALK_H__
#define __WALK_H__

#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <regex.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "err.h"

enum {
  WALK_OK = 0,
  WALK_BADPATTERN,
  WALK_NAMETOOLONG,
  WALK_BADIO,
};

#define WS_NONE0
#define WS_RECURSIVE (1 << 0)
#define WS_DEFAULT WS_RECURSIVE
#define WS_FOLLOWLINK (1 << 1)/* follow symlinks */
#define WS_DOTFILES (1 << 2)/* per unix convention, .file is hidden */
#define WS_MATCHDIRS (1 << 3)/* if pattern is used on dir names too */

inline int walk_recur(char *dname, regex_t *reg, int spec, std::vector<std::string> &dirs)
{
  struct dirent *dent;
  DIR *dir;
  struct stat st;
  char fn[FILENAME_MAX];
  int res = WALK_OK;
  int len = strlen(dname);
  if (len >= FILENAME_MAX - 1)
    return WALK_NAMETOOLONG;

  strcpy(fn, dname);
  fn[len++] = '/';

  if (!(dir = opendir(dname))) {
    return WALK_BADIO;
  }

  errno = 0;
  while ((dent = readdir(dir))) {
    if (!(spec & WS_DOTFILES) && dent->d_name[0] == '.')
      continue;
    if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, ".."))
      continue;

    strncpy(fn + len, dent->d_name, FILENAME_MAX - len);
    if (lstat(fn, &st) == -1) {
      res = WALK_BADIO;
      continue;
    }

    /* don't follow symlink unless told so */
    if (S_ISLNK(st.st_mode) && !(spec & WS_FOLLOWLINK))
      continue;

    /* will be false for symlinked dirs */
    if (S_ISDIR(st.st_mode)) {
      /* recursively follow dirs */
      if ((spec & WS_RECURSIVE))
	walk_recur(fn, reg, spec, dirs);

      if (!(spec & WS_MATCHDIRS)) continue;
    }

    /* pattern match */
    if (!regexec(reg, fn, 0, 0, 0)) dirs.push_back(std::string(fn));
  }

  if (dir) closedir(dir);
  return res ? res : errno ? WALK_BADIO : WALK_OK;
}

inline int walk_dir(char *dname, char *pattern, int spec, std::vector<std::string> &dirs)
{
  regex_t r;
  int res;
  if (regcomp(&r, pattern, REG_EXTENDED | REG_NOSUB))
    return WALK_BADPATTERN;
  res = walk_recur(dname, &r, spec, dirs);
  regfree(&r);

  return res;
}

inline std::vector<std::string> walk(std::string filename, std::string pattern = "", std::string filter = "")
{
  if (filename.size() && (filename[filename.size() - 1] == '/'))
    throw Err("don't end with slash");
  std::vector<std::string> files, files_tmp;
  
  int r = walk_dir(const_cast<char*>(filename.c_str()), const_cast<char*>(pattern.c_str()), WS_DEFAULT|WS_MATCHDIRS, files_tmp);
  switch(r) {
  case WALK_OK:break;
  case WALK_BADIO: throw Err("IO error");
  case WALK_BADPATTERN: throw Err("Bad pattern");
  case WALK_NAMETOOLONG: throw Err("Filename too long");
  default:
    throw Err("Unknown error?");
  }

  for (auto f : files_tmp)
    if (f.find(filter) != std::string::npos)
      files.push_back(f);
  return files;
}

#endif
