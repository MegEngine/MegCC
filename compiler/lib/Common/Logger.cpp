/**
 * \file compiler/lib/Common/Logger.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "compiler/Common/Logger.h"

using namespace megcc;

static LogLevel GlobalLogLevel = LogLevel::WARN;

void megcc::SetLogLevel(LogLevel level) {
    GlobalLogLevel = level;
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

// vim: syntax=cpp.doxygen
