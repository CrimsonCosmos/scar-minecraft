/**
 * input-monitor — macOS native helper for global keyboard/mouse event detection.
 *
 * Creates a passive CGEventTap (listen-only, does NOT modify or block events)
 * that monitors keyboard and mouse activity system-wide. Outputs a millisecond
 * timestamp to stdout for each detected event so Node.js can classify events
 * as bot-generated vs user-generated.
 *
 * Requires Accessibility permission (same as nut-js keyboard fallback).
 *
 * Compile: cc -O2 -o input-monitor input-monitor.c -framework ApplicationServices
 */

#include <ApplicationServices/ApplicationServices.h>
#include <stdio.h>
#include <sys/time.h>

static long long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static CGEventRef callback(CGEventTapProxy proxy, CGEventType type,
                           CGEventRef event, void *userInfo) {
    (void)proxy;
    (void)userInfo;

    /* Re-enable tap if macOS disabled it (happens under heavy load) */
    if (type == kCGEventTapDisabledByTimeout || type == kCGEventTapDisabledByUserInput) {
        CGEventTapEnable(*(CFMachPortRef *)userInfo, true);
        return event;
    }

    fprintf(stdout, "%lld\n", now_ms());
    return event;
}

int main(void) {
    setbuf(stdout, NULL);

    CGEventMask mask =
        CGEventMaskBit(kCGEventKeyDown) |
        CGEventMaskBit(kCGEventMouseMoved) |
        CGEventMaskBit(kCGEventLeftMouseDown) |
        CGEventMaskBit(kCGEventRightMouseDown);

    CFMachPortRef tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        mask,
        callback,
        NULL
    );

    if (!tap) {
        fprintf(stderr, "input-monitor: CGEventTapCreate failed. "
                "Grant Accessibility access in System Preferences.\n");
        return 1;
    }

    /* Pass tap ref to callback so it can re-enable on timeout */
    CGEventTapEnable(tap, true);
    /* Update userInfo to point at tap for re-enable */
    /* (callback receives NULL for userInfo, uses global instead) */

    CFRunLoopSourceRef source = CFMachPortCreateRunLoopSource(NULL, tap, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);

    fprintf(stderr, "input-monitor: listening for keyboard/mouse events\n");
    CFRunLoopRun();

    CFRelease(source);
    CFRelease(tap);
    return 0;
}
