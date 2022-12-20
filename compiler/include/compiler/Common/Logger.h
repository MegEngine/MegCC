/**
 * \file compiler/include/compiler/Common/Logger.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <iostream>
#include <vector>

namespace megcc {

enum class LogLevel : uint32_t {
    DEBUG = 0, /*!< The lowest level and most verbose */
    INFO = 1,  /*!< log information */
    WARN = 2,  /*!< Print only warning and errors */
    ERROR = 3, /*!< Print only errors */
};

void SetLogLevel(LogLevel);

void setAssertThrow(bool);

LogLevel GetLogLevel();

class Logger {
public:
    Logger(LogLevel level = LogLevel::DEBUG) : mCurrLevle(level) {}

    static Logger debug();
    static Logger info();
    static Logger warning();
    static Logger error();

    template <typename T>
    Logger& operator<<(const T& value) {
        if (mCurrLevle >= GetLogLevel()) {
            std::cerr << value << std::flush;
        }
        return *this;
    }

    template <typename T>
    Logger& operator<<(const std::vector<T>& value) {
        if (mCurrLevle >= GetLogLevel()) {
            std::cerr << "[";
            for (size_t i = 0; i < value.size(); i++) {
                if (i != 0)
                    std::cerr << ",";
                std::cerr << value[i];
            }
            std::cerr << "]" << std::flush;
        }
        return *this;
    }

private:
    LogLevel mCurrLevle = LogLevel::DEBUG;
};

class LogFatal : public Logger {
public:
    LogFatal();
    ~LogFatal();
};

#define LOG_DEBUG megcc::Logger::debug()
#define LOG_INFO megcc::Logger::info()
#define LOG_WARN megcc::Logger::warning()
#define LOG_ERROR megcc::Logger::error()

#define CC_ASSERT(expr)                                                     \
    if (!(expr))                                                            \
    megcc::LogFatal() << "MegCC Assert " << #expr                           \
                      << " Error, in file: " << __FILE__ << ":" << __LINE__ \
                      << ", at function: " << __PRETTY_FUNCTION__           \
                      << ", line: " << __LINE__ << ". extra message: "

//! in order to destruct LogFatal after message is print, add `if(true)`
#define CC_ABORT                                                              \
    if (true)                                                                 \
    megcc::LogFatal() << "MegCC Abort, Error in file: " << __FILE__ << ":"    \
                      << __LINE__ << ", at function: " << __PRETTY_FUNCTION__ \
                      << ", line: " << __LINE__ << ". extra message: "

}  // namespace megcc

// vim: syntax=cpp.doxygen
