#pragma once

// ANSI color codes for terminal output
namespace Color {
    inline constexpr const char* RESET   = "\033[0m";
    inline constexpr const char* RED     = "\033[31m";
    inline constexpr const char* GREEN   = "\033[32m";
    inline constexpr const char* YELLOW  = "\033[33m";
    inline constexpr const char* BLUE    = "\033[34m";
    inline constexpr const char* MAGENTA = "\033[35m";
    inline constexpr const char* CYAN    = "\033[36m";
    inline constexpr const char* ORANGE  = "\033[38;5;208m";  // Bright orange (256-color mode)
    inline constexpr const char* BOLD    = "\033[1m";
}
