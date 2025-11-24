## Git 설치 (apt 이용)
```bash
sudo apt update
sudo apt install git
```

---

## SSH 키 생성
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

---

## pip 캐시 디렉토리 변경

1. **pip.conf 파일 생성/수정**
    ```bash
    mkdir -p ~/.pip
    echo "[global]
    cache-dir = /path/to/new/cache/dir" > ~/.pip/pip.conf
    ```

2. **현재 세션에만 적용**
    ```bash
    export PIP_CACHE_DIR=/path/to/new/cache/dir
    ```

3. **영구 적용 (bashrc에 추가)**
    ```bash
    echo 'export PIP_CACHE_DIR=/path/to/new/cache/dir' >> ~/.bashrc
    source ~/.bashrc
    ```

4. **pip 캐시 위치 확인**
    ```bash
    pip cache dir
    ```

5. **캐시 크기 확인**
    ```bash
    du -sh $(pip cache dir)
    ```

6. **기존 캐시 이동**
    ```bash
    mv ~/.cache/pip/* /path/to/new/cache/dir/
    ```

7. **(선택) 심볼릭 링크 생성**
    ```bash
    ln -s /path/to/new/cache/dir ~/.cache/pip
    ```

---
