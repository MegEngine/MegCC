#include "compiler/Common/Logger.h"
#include <iostream>

using namespace megcc;

static LogLevel GlobalLogLevel = LogLevel::WARN;

static bool g_is_assert_throw = false;

void megcc::SetLogLevel(LogLevel level) {
    GlobalLogLevel = level;
}

void megcc::setAssertThrow(bool is_throw) {
    g_is_assert_throw = is_throw;
}

LogLevel megcc::GetLogLevel() {
    return GlobalLogLevel;
}

Logger Logger::debug() {
    return Logger(LogLevel::DEBUG);
}

Logger Logger::info() {
    return Logger(LogLevel::INFO);
}

Logger Logger::warning() {
    return Logger(LogLevel::WARN);
}

Logger Logger::error() {
    return Logger(LogLevel::ERROR);
}

LogFatal::LogFatal() : Logger(LogLevel::ERROR) {
#if __EXCEPTIONS
    if (g_is_assert_throw) {
        throw std::exception();
    }
#endif
}
LogFatal::~LogFatal() {
    abort();
}

// vim: syntax=cpp.doxygen
