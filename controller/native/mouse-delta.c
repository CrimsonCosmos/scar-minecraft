/**
 * mouse-delta — macOS native helper for relative mouse movement.
 *
 * Reads "dx dy\n" from stdin, posts CGEvent mouse-moved events with
 * delta fields set so games reading raw input through pointer lock
 * receive the movement. Stays alive as a persistent subprocess for
 * zero-latency input (~50μs per call, no spawn overhead).
 *
 * Compile: cc -O2 -o mouse-delta mouse-delta.c -framework ApplicationServices
 */

#include <ApplicationServices/ApplicationServices.h>
#include <stdio.h>

int main(void) {
    int dx, dy;
    setbuf(stdout, NULL);

    while (scanf("%d %d", &dx, &dy) == 2) {
        CGEventRef cur = CGEventCreate(NULL);
        CGPoint pos = CGEventGetLocation(cur);
        CFRelease(cur);

        CGEventRef move = CGEventCreateMouseEvent(
            NULL, kCGEventMouseMoved,
            CGPointMake(pos.x + dx, pos.y + dy),
            kCGMouseButtonLeft
        );
        CGEventSetIntegerValueField(move, kCGMouseEventDeltaX, dx);
        CGEventSetIntegerValueField(move, kCGMouseEventDeltaY, dy);
        CGEventPost(kCGHIDEventTap, move);
        CFRelease(move);

        printf("ok\n");
    }

    return 0;
}
