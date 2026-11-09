// usage: busyleader FIFO CTRFILE
// Main thread blocks SIGCHLD, starts 8 busy (R) workers that inherit the mask, then
// posix_spawns a child that blocks opening FIFO before exec. Worker 0 publishes a counter.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static int ctr_fd;

static void *worker(void *arg) {
    volatile uint64_t n = 0;
    for (;;) {
        n++;
        if (arg == NULL && (n & 0xfffff) == 0) {
            uint64_t v = n >> 20;
            if (pwrite(ctr_fd, &v, sizeof v, 0) < 0) _exit(9);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    ctr_fd = open(argv[2], O_RDWR | O_CREAT, 0600);
    sigset_t m;
    sigemptyset(&m);
    sigaddset(&m, SIGCHLD);
    pthread_sigmask(SIG_BLOCK, &m, NULL);
    pthread_t w;
    for (long i = 0; i < 8; i++) pthread_create(&w, NULL, worker, (void *)i);
    usleep(20000);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, argv[1], O_RDONLY, 0);
    char *cargv[] = {"true", NULL};
    char *envp[] = {NULL};
    pid_t pid;
    if (posix_spawn(&pid, "/bin/true", &fa, NULL, cargv, envp)) return 3;
    int st;
    waitpid(pid, &st, 0);
    return 0;
}
