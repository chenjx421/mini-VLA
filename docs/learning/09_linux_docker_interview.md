# Linux 与 Docker 三天冲刺学习文档

目标：三天后能够看懂常见 Linux 与 Docker 命令，完成基础排障，并回答秋招笔试和面试中的高频问题。

适用环境：

    Mac
      |
      +-- SSH --> Ubuntu/Jetson 采集板
                      |
                      +-- Docker --> 采集容器

每天投入 3 到 4 小时，按照理解、实操、排障、面试题的顺序学习。

## 环境边界

    Mac 提示符：       jiaxin.chen@MacBook-Air ~ %
    Linux 宿主机：     user@edge-device:~$
    Docker 容器：      root@8a91bc234def:/#

    ssh、adb               通常在 Mac 执行
    systemctl、journalctl  通常在 Linux 宿主机执行
    docker ps、docker logs 在 Docker 宿主机执行
    ls、ps、cat            作用于当前所在的层

提示符中的百分号、美元符号、井号不要输入。每条命令先确认自己在哪一层。

---

# 第一天：命令、文件、文本、权限和进程

## 学习目标

- 拆解陌生命令的命令名、选项和参数
- 看懂绝对路径和相对路径
- 查找文件和大文件
- 使用管道、重定向和 grep 分析日志
- 理解权限和 sudo
- 查看进程并解释退出码

## 1. 命令结构

    command [options] [arguments]

例子：

    ls -lah /var/log

    ls       命令：列出目录
    -lah     选项：详细、显示隐藏文件、使用易读单位
    /var/log 参数：要查看的目录

短选项可以合并：

    ls -l -a -h
    ls -lah

长选项通常使用两个横线：

    docker ps --all

## 2. 退出码和常见错误

    pwd
    echo $?

上一条命令的退出码通常是：

    0       成功
    非 0    失败或异常

常见错误：

    command not found
        命令未安装，或命令不在 PATH 中。

    No such file or directory
        路径不存在、拼写错误，或当前机器不对。

    Permission denied
        权限不足，或脚本没有执行权限。

    Connection refused
        目标可达，但端口没有服务监听或被拒绝。

    Operation timed out
        在规定时间内没有网络回应。

    Address already in use
        端口已被其他进程占用。

## 3. 确认当前环境

    whoami
    hostname
    uname -r
    uname -m
    pwd

    whoami   当前用户名
    hostname 设备名
    uname -r 当前内核版本
    uname -m CPU 架构，例如 aarch64
    pwd      当前工作目录

Mac 上 uname -s 通常输出 Darwin；Linux 宿主机和容器通常输出 Linux。

## 4. 路径和目录

    /       根目录
    ~       当前用户家目录
    .       当前目录
    ..      上一级目录

    pwd
    ls
    ls -la
    cd /data
    cd ..
    cd ~
    cd -

绝对路径从根目录开始，例如 /data/capture；相对路径从当前目录开始，例如 logs。

## 5. 文件操作练习

先在自己的练习目录中执行：

    mkdir -p ~/linux-practice
    cd ~/linux-practice
    touch app.log
    echo "INFO capture started" > app.log
    echo "ERROR camera timeout" >> app.log
    cat app.log
    cp app.log app.log.bak
    mv app.log.bak backup.log
    ls -l

    mkdir -p  创建目录
    touch     创建空文件或更新时间
    >         覆盖写入
    >>        追加写入
    cat       输出文件内容
    cp        复制
    mv        移动或重命名

rm 会直接删除文件。不要在不确认路径时使用递归强制删除。

## 6. 查找文件

    find /data/capture -type f -name 'sync_gpio.ko'
    find /data/capture -type f -name '*.log' -mtime -1
    find /data/capture -type f -size +100M -ls

    find               递归查找
    -type f            只找普通文件
    -name              按名称匹配
    -mtime -1          最近一天修改
    -size +100M        大于 100 MB

## 7. 管道、文本和重定向

    ps aux | grep GLZN_CAPTURE_APP
    head -n 20 app.log
    tail -n 50 app.log
    tail -f app.log
    grep -i 'error' app.log
    grep -n 'timeout' app.log
    wc -l app.log
    sort app.log | uniq -c

    |       管道，把左侧输出交给右侧
    head    查看开头
    tail    查看结尾
    tail -f 实时跟踪新增日志
    grep    筛选匹配行
    -i      忽略大小写
    -n      显示行号
    wc -l   统计行数

    ps aux > processes.txt
    date >> check.log
    ls /not-exist 2> error.log
    command > all.log 2>&1

    0   标准输入
    1   标准输出
    2   标准错误

## 8. 权限

    ls -l startup.sh

示例：

    -rwxr-x--- 1 user capture 532 Aug 18 10:20 startup.sh

    rwx r-x ---
    所有者 同组 其他用户

    r = 4    读
    w = 2    写
    x = 1    执行

    755 = rwx r-x r-x
    644 = rw- r-- r--
    700 = rwx --- ---

    chmod +x startup.sh
    chmod 755 startup.sh
    sudo chown user:user startup.sh

## 9. 进程

    ps aux
    ps aux | grep GLZN_CAPTURE_APP
    top
    ps -ef | grep my-process

重点字段：

    PID       进程 ID
    %CPU      CPU 使用率
    %MEM      内存使用率
    STAT      进程状态
    COMMAND   启动命令

结束进程：

    kill PID
    kill -9 PID

先使用普通 kill。kill -9 不给程序清理资源的机会，只在普通 kill 无效时使用。

## Day 1 实战

    LOG_DIR=/data/capture/logs
    echo "日志目录：$LOG_DIR"
    find "$LOG_DIR" -type f -name '*.log' -mtime -1
    grep -inE 'error|failed|timeout|exception' "$LOG_DIR"/*.log 2>/dev/null | tail -n 50

要求：解释变量、双引号、管道、-i、-n、-E 和 2>/dev/null。

第一天面试题：

1. > 和 >> 有什么区别？
2. 2>&1 的含义是什么？
3. 相对路径和绝对路径有什么区别？
4. chmod 755 和 chmod 644 的区别是什么？
5. kill 和 kill -9 的区别是什么？
6. 如何找出 /data 下大于 1GB 的文件？

---

# 第二天：服务、日志、磁盘和网络

## 学习目标

- 判断 systemd 服务是正常、停止还是启动失败
- 从日志中定位真正的失败原因
- 判断 CPU、内存、磁盘和 inode 是否异常
- 判断端口是否监听、网络是否可达
- 建立固定的故障排查顺序

## 1. systemd 服务

    systemctl status capture.service --no-pager
    systemctl is-active capture.service
    systemctl is-enabled capture.service
    systemctl cat capture.service

状态含义：

    active (running)  正在运行
    inactive (dead)   没有运行
    failed            启动或运行失败
    activating        正在启动

重启服务有副作用：

    sudo systemctl restart capture.service

工作设备上执行前，先确认是否会中断采集。

## 2. journalctl

    journalctl -u capture.service -n 100 --no-pager
    journalctl -u capture.service -b --no-pager
    journalctl -u capture.service -f
    journalctl -p err -b --no-pager

    -u service   指定服务
    -n 100       最近 100 行
    -b           本次启动
    -f           实时跟踪
    -p err       错误级别
    --no-pager   不进入翻页器

筛选错误：

    journalctl -u capture.service -n 200 --no-pager | grep -iE 'error|failed|timeout|permission|not found'

## 3. CPU、内存和负载

    top
    free -h
    uptime
    nproc

free -h 优先关注 available。Linux 会使用空闲内存做缓存，free 小不一定表示内存不足。

uptime 的 load average 是 1、5、15 分钟平均负载，要结合 CPU 核数判断是否过高。

## 4. 磁盘和 inode

    df -h
    df -h /data
    df -ih
    du -sh /data/capture
    du -h --max-depth=1 /data/capture | sort -h

    df      查看文件系统整体剩余空间
    du      统计目录和文件占用空间
    df -i   查看 inode 使用情况

如果 df 很高但 du 对不上，检查：

    sudo lsof +L1

常见原因是文件已经删除，但仍被进程打开。

## 5. 网络基础

    ip addr
    ip route
    ping -c 4 192.0.2.10
    curl -I http://192.0.2.10:8080
    ss -lntp
    nc -vz -w 2 192.0.2.10 5555

ss 选项：

    -l  监听中
    -n  数字显示地址和端口
    -t  TCP
    -p  显示进程

网络排障顺序：

    IP 是否正确
    设备是否在线
    路由是否存在
    端口是否监听
    服务是否接受连接
    应用协议和认证是否正确

Mac 的 nc 常见 -G 2，Linux 常见 -w 2，不要机械混用。

## Day 2 实战：采集服务故障排查

    systemctl is-active capture.service
    systemctl status capture.service --no-pager
    journalctl -u capture.service -n 100 --no-pager
    docker ps -a
    docker stats --no-stream
    df -h /data
    df -ih /data
    du -h --max-depth=1 /data | sort -h | tail -n 20
    ss -lntp

判断逻辑：

    服务 failed
      -> journalctl 看第一条失败原因

    服务 active，但容器 exited
      -> docker logs 看容器退出原因

    容器运行，但程序无响应
      -> docker exec 进入容器查进程和文件

    日志写不进去
      -> df、df -i、权限、挂载点

    设备访问不了
      -> ip、route、ping、ss、nc

第二天面试题：

1. systemctl status 和 journalctl 分别解决什么问题？
2. df 和 du 的区别是什么？
3. 什么是 inode？inode 用完会发生什么？
4. active (running) 是否代表业务一定正常？
5. Connection refused 和 Operation timed out 的区别是什么？
6. 如何查 8080 端口被谁占用？

---

# 第三天：Docker、Shell 和综合面试

## 学习目标

- 说清楚镜像、容器、仓库的关系
- 熟练使用 ps、logs、inspect、exec、stats
- 看懂 Dockerfile 基本指令
- 理解端口、挂载、网络和容器生命周期
- 写简单的 Shell 检查脚本
- 完成一轮综合面试题

## 1. Docker 三个核心对象

    Dockerfile --build--> Image --run--> Container
                              |
                              +-- push/pull --> Registry

    Image      只读模板
    Container  镜像运行后的实例
    Registry   保存和分发镜像的服务器

查看：

    docker image ls
    docker ps
    docker ps -a

镜像可以存在但没有运行；容器可以存在但已经停止；docker ps 默认只显示运行中的容器。

## 2. 容器生命周期

    docker create --name test ubuntu:22.04
    docker start test
    docker stop test
    docker rm test

    create  创建但不启动
    start   启动已有容器
    stop    请求程序停止
    rm      删除已停止容器

    docker run --name test ubuntu:22.04 echo hello

这个容器执行完 echo hello 就会退出，因为容器是否运行取决于主进程是否还在运行。

## 3. 容器排查

    docker ps -a
    docker inspect capture_container
    docker logs --tail 100 capture_container
    docker stats --no-stream capture_container
    docker exec -it capture_container bash
    docker exec -it capture_container sh

exec 在容器中启动新进程，适合排障；attach 连接主进程，退出方式不当可能影响主进程。

只执行一次命令：

    docker exec capture_container ps aux
    docker exec capture_container pwd

常查 inspect 字段：

    .Config.Image
    .Image
    .State.Status
    .State.ExitCode
    .Mounts
    .NetworkSettings.Ports
    .Config.Env

格式化示例：

    docker inspect capture_container --format 'Status={{.State.Status}} ExitCode={{.State.ExitCode}}'

## 4. Dockerfile

    FROM ubuntu:22.04
    WORKDIR /app
    COPY requirements.txt .
    RUN apt-get update && apt-get install -y python3
    COPY . .
    EXPOSE 8080
    ENTRYPOINT ["python3"]
    CMD ["server.py"]

    FROM        基础镜像
    WORKDIR     工作目录
    COPY        复制文件
    RUN         构建镜像时执行
    EXPOSE      声明端口，不等于映射
    ENTRYPOINT  固定入口程序
    CMD         默认命令或参数

默认执行：

    python3 server.py

## 5. 端口和挂载

    docker run -p 8080:80 nginx

表示：

    宿主机 8080  -->  容器 80

目录挂载：

    docker run -v /data/capture:/app/data image-name

表示：

    宿主机 /data/capture  -->  容器 /app/data

容器删除后，容器可写层可能丢失；宿主机目录挂载和 volume 中的数据通常保留。

## 6. Shell 检查脚本

    #!/usr/bin/env bash
    set -u
    CONTAINER="capture_container"

    echo "===== 基础信息 ====="
    date '+%F %T'
    hostname
    uname -r
    uname -m

    echo "===== 容器状态 ====="
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "container=$CONTAINER status=running"
    else
      echo "container=$CONTAINER status=not-running"
    fi

    echo "===== 磁盘空间 ====="
    df -h /data

检查语法：

    bash -n check.sh

执行：

    bash check.sh
    chmod +x check.sh
    ./check.sh

## Day 3 实战：判断容器为何退出

    CONTAINER=capture_container
    docker ps -a --filter "name=$CONTAINER"
    docker inspect "$CONTAINER" --format 'Status={{.State.Status}} ExitCode={{.State.ExitCode}} Error={{.State.Error}}'
    docker logs --tail 200 "$CONTAINER"

    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      docker exec "$CONTAINER" ps aux
      docker exec "$CONTAINER" df -h
    fi

判断：

    ExitCode=0
      可能是主程序正常执行完毕，容器没有常驻进程
    ExitCode 非 0
      程序异常退出，优先看 docker logs
    Error 不为空
      Docker 创建或启动容器本身失败
    容器运行但业务异常
      检查进程、文件、环境变量、挂载和网络

第三天面试题：

1. 镜像和容器有什么区别？
2. 为什么容器启动后立即退出？
3. docker exec 和 docker attach 有什么区别？
4. EXPOSE 是否会自动映射端口？
5. 容器删除后哪些数据可能丢失？
6. CMD 和 ENTRYPOINT 有什么区别？
7. 为什么 Dockerfile 要把不常变化的步骤放在前面？
8. 容器内的 localhost 指向谁？

---

# 高频面试答案模板

## 服务启动失败如何排查？

    先用 systemctl status 查看服务状态和 Main PID；
    再用 journalctl -u 查看日志，定位第一条真正的失败原因；
    如果服务负责启动 Docker，再检查 docker ps -a 和 docker logs；
    最后检查权限、文件路径、端口、挂载和磁盘空间。

## Docker 容器启动后立即退出怎么办？

    先 docker ps -a 看状态和退出码；
    再 docker inspect 看 State.Error、Path、Args；
    用 docker logs 查看应用输出；
    如果容器能保持运行，再用 docker exec 查看进程、文件和环境；
    常见原因是主进程执行完毕、配置错误、权限错误或依赖不可用。

## 如何排查端口被占用？

    ss -lntp | grep ':8080'
    ps -fp PID

先确认监听进程，再决定是否停止；不要直接杀进程。

## Linux 磁盘满了怎么办？

    先 df -h 判断哪个文件系统满了；
    再 df -i 判断 inode 是否用完；
    用 du -xhd1 找出大目录；
    检查日志、Docker 镜像、容器可写层和已删除但仍被打开的文件；
    确认内容后再清理，避免直接递归强制删除。

## 为什么服务 active 但业务不正常？

    active 只表示 systemd 服务主进程仍存在；
    不代表内部线程、子进程、相机设备、网络连接和数据链路都正常；
    还要结合进程、日志、端口、设备节点和实际输出判断。

---

# 三天每日验收

## 第一天

不看资料完成并解释：

    pwd
    find /data -type f -name '*.log'
    grep -in 'error' app.log | tail -n 20
    df -h /data
    ps aux | grep capture

## 第二天

回答并操作：

    服务 failed 时第一步看什么？
    端口不通如何区分拒绝和超时？
    磁盘空间和 inode 如何分别检查？
    如何实时查看服务日志？

## 第三天

完成：

    查看容器是否存在
    查看容器退出码
    查看容器日志
    查看容器镜像和挂载
    进入容器检查进程
    解释容器为什么退出

---

# 最后速查表

    # 环境
    whoami
    hostname
    uname -a
    pwd

    # 文件
    ls -lah
    cd /path
    find /path -type f -name '*.log'
    cp source target
    mv old new

    # 文本
    cat file
    head -n 20 file
    tail -f file
    grep -in 'error' file
    sort file | uniq -c

    # 权限
    ls -l file
    chmod 755 script.sh
    sudo command

    # 进程
    ps aux
    top
    kill PID

    # 服务和日志
    systemctl status service --no-pager
    journalctl -u service -n 100 --no-pager

    # 磁盘
    df -h
    df -ih
    du -sh path

    # 网络
    ip addr
    ip route
    ss -lntp
    ping -c 4 host
    curl -I URL
    nc -vz -w 2 host port

    # Docker
    docker images
    docker ps -a
    docker inspect container
    docker logs --tail 100 container
    docker exec -it container bash
    docker stats --no-stream

每条命令都回答四个问题：

    它在哪台机器执行？
    它读取还是修改数据？
    输出的关键字段是什么？
    失败后下一步看哪里？
