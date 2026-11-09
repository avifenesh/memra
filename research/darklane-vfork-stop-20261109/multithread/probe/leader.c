// Job leader for the PR #1130 review probe.
// usage: leader MODE FIFO CTRFILE
//   single        main thread posix_spawns a child that blocks opening FIFO before exec
//   main-spawn    same, plus a worker thread that counts (nanosleep 1 ms per step)
//   thread-spawn  a spawner thread posix_spawns; main joins it (S); worker counts (PR test shape)
//   main-spawn-nocldstop  main-spawn with a SIGCHLD handler installed with SA_NOCLDSTOP
//   main-spawn-maskchld   main-spawn with the worker thread blocking SIGCHLD
//   two-block     main posix_spawns the FIFO child; the worker first sits in an unrelated
//                 uninterruptible wait (vfork on a child that leaves the group, sleeps 1.5 s
//                 and execs), then counts
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static const char *fifo;
static int ctr_fd;
static int worker_vfork_first;
static int worker_mask_chld;

static void on_chld(int sig) { (void)sig; }

static void spawn_fifo_child(void) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, fifo, O_RDONLY, 0);
    char *argv[] = {"true", NULL};
    char *envp[] = {NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, "/bin/true", &fa, NULL, argv, envp);
    if (rc) { fprintf(stderr, "posix_spawn %d\n", rc); exit(2); }
    int st;
    waitpid(pid, &st, 0);
}

static void *worker(void *arg) {
    (void)arg;
    if (worker_mask_chld) {
        sigset_t m;
        sigemptyset(&m);
        sigaddset(&m, SIGCHLD);
        pthread_sigmask(SIG_BLOCK, &m, NULL);
    }
    if (worker_vfork_first) {
        // An unrelated uninterruptible wait that ends without any signal to this process:
        // the child leaves the job's group (the group stop never reaches it), sleeps 1.5 s,
        // then execs, which releases the vfork wait (no SIGCHLD: the child has not exited).
        pid_t c = vfork();
        if (c == 0) {
            setpgid(0, 0);
            struct timespec ts = {1, 500000000};
            nanosleep(&ts, NULL);
            execl("/bin/sleep", "sleep", "30", (char *)NULL);
            _exit(127);
        }
    }
    uint64_t n = 0;
    struct timespec ts = {0, 1000000};
    for (;;) {
        n++;
        pwrite(ctr_fd, &n, sizeof n, 0);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void *spawner(void *arg) {
    (void)arg;
    spawn_fifo_child();
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const char *mode = argv[1];
    fifo = argv[2];
    ctr_fd = open(argv[3], O_RDWR | O_CREAT, 0600);
    pthread_t w, s;
    if (!strcmp(mode, "single")) {
        spawn_fifo_child();
    } else if (!strcmp(mode, "main-spawn")) {
        pthread_create(&w, NULL, worker, NULL);
        usleep(20000);
        spawn_fifo_child();
    } else if (!strcmp(mode, "main-spawn-nocldstop")) {
        // A SIGCHLD handler installed with SA_NOCLDSTOP (the usual flag for one).
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = on_chld;
        sa.sa_flags = SA_NOCLDSTOP | SA_RESTART;
        sigaction(SIGCHLD, &sa, NULL);
        pthread_create(&w, NULL, worker, NULL);
        usleep(20000);
        spawn_fifo_child();
    } else if (!strcmp(mode, "main-spawn-maskchld")) {
        // The other thread blocks SIGCHLD, as helper threads that block signals do.
        worker_mask_chld = 1;
        pthread_create(&w, NULL, worker, NULL);
        usleep(20000);
        spawn_fifo_child();
    } else if (!strcmp(mode, "thread-spawn")) {
        pthread_create(&w, NULL, worker, NULL);
        pthread_create(&s, NULL, spawner, NULL);
        pthread_join(s, NULL);
    } else if (!strcmp(mode, "two-block") || !strcmp(mode, "two-block-nocldstop")) {
        if (!strcmp(mode, "two-block-nocldstop")) {
            struct sigaction sa;
            memset(&sa, 0, sizeof sa);
            sa.sa_handler = on_chld;
            sa.sa_flags = SA_NOCLDSTOP | SA_RESTART;
            sigaction(SIGCHLD, &sa, NULL);
        }
        worker_vfork_first = 1;
        pthread_create(&w, NULL, worker, NULL);
        usleep(50000);
        spawn_fifo_child();
    } else {
        return 2;
    }
    return 0;
}
