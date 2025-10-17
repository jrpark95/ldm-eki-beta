////////////////////////////////////////////////////////////////////////////////
/// @file colors.h
/// @brief ANSI color codes for terminal output formatting
/// @author Juryong Park
/// @date 2025
///
/// Provides a centralized namespace of ANSI escape sequences for colorful and
/// formatted terminal output. Used throughout the application for consistent
/// visual feedback (errors in red, success in green, headers in cyan, etc.).
///
/// @note These codes are portable across ANSI-compatible terminals. Non-ANSI
///       terminals will display the codes as text, but functionality is not
///       affected.
////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace Color {
    /// Reset all formatting to default
    inline constexpr const char* RESET   = "\033[0m";

    /// Standard colors
    inline constexpr const char* RED     = "\033[31m";
    inline constexpr const char* GREEN   = "\033[32m";
    inline constexpr const char* YELLOW  = "\033[33m";
    inline constexpr const char* BLUE    = "\033[34m";
    inline constexpr const char* MAGENTA = "\033[35m";
    inline constexpr const char* CYAN    = "\033[36m";

    /// Extended 256-color palette (bright orange for milestone events)
    inline constexpr const char* ORANGE  = "\033[38;5;208m";

    /// Text formatting
    inline constexpr const char* BOLD    = "\033[1m";
}
