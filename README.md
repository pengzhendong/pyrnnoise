# pyrnnoise

```bash
$ pip install pyrnnoise
$ denoise in.wav out.wav --plot
```

## Build

``` bash
$ git submodule update --init
$ cmake -B pyrnnoise/build -DCMAKE_BUILD_TYPE=Release
$ cmake --build pyrnnoise/build --target install
$ pip install -e .
```
