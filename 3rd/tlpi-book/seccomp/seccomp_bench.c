/*************************************************************************\
*                  Copyright (C) Michael Kerrisk, 2022.                   *
*                                                                         *
* This program is free software. You may use, modify, and redistribute it *
* under the terms of the GNU General Public License as published by the   *
* Free Software Foundation, either version 3 or (at your option) any      *
* later version. This program is distributed without any warranty.  See   *
* the file COPYING.gpl-v3 for details.                                    *
\*************************************************************************/

/* Supplementary program for Chapter Z */

#define _GNU_SOURCE
#include <sys/syscall.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define errExit(msg)    do { perror(msg); exit(EXIT_FAILURE); \
                        } while (0)

static int
seccomp(unsigned int operation, unsigned int flags, void *arg)
{
    return syscall(__NR_seccomp, operation, flags, arg);
    // Or: return prctl(PR_SET_SECCOMP, operation, arg);
}

static void
install_filter(char *instr, int icnt)
{
    struct sock_filter load = BPF_STMT(BPF_LD | BPF_W | BPF_ABS, 0);
    struct sock_filter jump = BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 0);
    struct sock_filter add = BPF_STMT(BPF_ALU | BPF_ADD | BPF_K, 1);
    struct sock_filter ret = BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    struct sock_filter instruction;
    struct sock_filter *filter;

    filter = calloc(icnt + 1, sizeof(struct sock_filter));

    /* Create a filter containing 'icnt' instructions of the kind specified
       in 'instr' */

    if (instr[0] == 'a')
        instruction = add;
    else if (instr[0] == 'j')
        instruction = jump;
    else if (instr[0] == 'l')
        instruction = load;
    else {
        fprintf(stderr, "Bad instruction value: %s\n", instr);
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < icnt; j++)
        filter[j] = instruction;

    /* Add a return instruction to terminate the filter */

    filter[icnt] = ret;

    /* Install the BPF filter */

    struct sock_fprog prog = {
        .len = icnt + 1,
        .filter = filter,
    };

    if (seccomp(SECCOMP_SET_MODE_FILTER, 0, &prog) == -1)
        errExit("seccomp");
}

int
main(int argc, char *argv[])
{
    if (argc != 2 && argc < 4) {
        fprintf(stderr, "Usage: %s <num-loops> [<add|jump|load> "
                "<instr-cnt> [num-filters]]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (argc >= 4) {
        int nfilters = (argc > 4) ? atoi(argv[4]) : 1;
        int icnt = atoi(argv[3]);

        printf("Applying BPF filter\n");

        if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
            errExit("prctl");

        for (int j = 0; j < nfilters; j++)
            install_filter(argv[2], icnt);
    }

    int nloops = atoi(argv[1]);

    for (int j = 0; j < nloops; j++)
        getppid();

    exit(EXIT_SUCCESS);
}
