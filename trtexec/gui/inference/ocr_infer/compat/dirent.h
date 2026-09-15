#pragma once

#ifdef _WIN32

#include <windows.h>

#include <cstring>
#include <string>

#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif

struct dirent {
    char d_name[PATH_MAX];
};

struct DIR {
    HANDLE handle = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATAA data{};
    dirent entry{};
    bool first = true;
};

inline DIR* opendir(const char* path)
{
    if (!path || !*path) {
        return nullptr;
    }
    std::string pattern(path);
    const char last = pattern.back();
    if (last != '/' && last != '\\') {
        pattern += "\\";
    }
    pattern += "*";

    auto* dir = new DIR();
    dir->handle = FindFirstFileA(pattern.c_str(), &dir->data);
    if (dir->handle == INVALID_HANDLE_VALUE) {
        delete dir;
        return nullptr;
    }
    return dir;
}

inline dirent* readdir(DIR* dir)
{
    if (!dir || dir->handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    if (!dir->first) {
        if (!FindNextFileA(dir->handle, &dir->data)) {
            return nullptr;
        }
    }
    dir->first = false;
    std::strncpy(dir->entry.d_name, dir->data.cFileName, PATH_MAX - 1);
    dir->entry.d_name[PATH_MAX - 1] = '\0';
    return &dir->entry;
}

inline int closedir(DIR* dir)
{
    if (!dir) {
        return -1;
    }
    if (dir->handle != INVALID_HANDLE_VALUE) {
        FindClose(dir->handle);
    }
    delete dir;
    return 0;
}

#else
#include_next <dirent.h>
#endif
