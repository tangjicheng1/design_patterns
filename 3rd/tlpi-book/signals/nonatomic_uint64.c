/*************************************************************************\
*                  Copyright (C) Michael Kerrisk, 2022.                   *
*                                                                         *
* This program is free software. You may use, modify, and redistribute it *
* under the terms of the GNU General Public License as published by the   *
* Free Software Foundation, either version 3 or (at your option) any      *
* later version. This program is distributed without any warranty.  See   *
* the file COPYING.gpl-v3 for details.                                    *
\*************************************************************************/

/* Supplementary program for Chapter 21 */

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#define errExit(msg)    do { perror(msg); exit(EXIT_FAILURE); } while (0)

#define SIG SIGUSR1

static volatile uint64_t ip = 0;  /* 'volatile' included here just to show
                                      that it really doesn't help. */

/* Signal handler inverts bits in '*ip'; this may not be atomic. */

static void
handler(int sig)
{
    ip = ~ip;
}

int
main(int argc, char *argv[])
{
    /* Set up signal handler. */

    struct sigaction sa;

    sa.sa_handler = handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIG, &sa, NULL) == -1)
        errExit("sigaction");

    /* Create child that will blast signals at parent. */

    pid_t childPid = fork();
    if (childPid == -1)
        errExit("fork");

    if (childPid == 0) {
        while (1) {
            if (kill(getppid(), SIG) == -1)
                errExit("kill() failed in child");
        }
    }

    /* Parent falls through */

    /* Loop, fetching the value of '*ip' (the fetch may not be atomic)
       and checking whether the result is all bits zero or all bits one.
       If it is not, print the fetched value and the loop counter. */

    for (int long long j = 0; ; j++) {
        uint64_t loc = ip;
        if (loc != 0 && loc != ~0)
            printf("Unexpected: %016" PRIx64 " (loop %lld)\n", loc, j);
    }

    exit(EXIT_SUCCESS);
}
