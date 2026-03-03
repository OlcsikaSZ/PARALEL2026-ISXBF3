# Game of Life (Életjáték) – OpenCL alapú párhuzamos megvalósítás (terv / koncepció)

Ez a beadandó célja egy **Conway-féle Game of Life** (Életjáték) szimulátor elkészítése **OpenCL** technológiával. A projekt fókusza nem csak a működő szimuláció lesz, hanem az, hogy a megoldás **párhuzamos** legyen, **mérhető** legyen, és a teljesítmény **tudatosan hangolható** legyen (tuning).

A dokumentum jelenlegi formájában **témabemutató jellegű**: azt írja le, **mi fog elkészülni**, milyen modulokból fog állni, és hogyan fogjuk igazolni a megoldás helyességét és gyorsaságát.

---

## 1. A feladat röviden

Az Életjáték egy kétdimenziós rácson futó diszkrét szimuláció, ahol a cellák két állapotúak:

- **élő (1)**
- **halott (0)**

Minden lépésben (iterációban) minden cella új állapota a 8 szomszéd alapján számolódik:

- Élő sejt **túlél**, ha 2 vagy 3 élő szomszédja van, különben **elpusztul**.
- Halott sejt **életre kel**, ha pontosan 3 élő szomszédja van.

A cél az, hogy ezt **OpenCL kernellel** számoljuk, és a rács méretének növelésével megmutassuk, hogy a párhuzamos végrehajtás miért előnyös.

---

## 2. Tervezett célok és követelmények

### 2.1 Funkcionális célok (mit fog tudni a program)

A kész program várhatóan képes lesz:

- tetszőleges méretű rács kezelésére (`rows`, `cols`)
- megadott iterációszám futtatására (`iters`)
- kezdőállapot generálására (pl. random seed alapján)
- határfeltételek kezelésére:
  - **wrap = 0:** fix perem (a rácson kívüli cellák halottnak számítanak)
  - **wrap = 1:** toroid (körbeérő rács)
- több kernel-variáns közti váltásra (naiv vs optimalizált)
- mérési eredmények kiírására (konzol + opcionális CSV)

### 2.2 Nem-funkcionális célok (minőség, mérhetőség)

A beadandó célja, hogy:

- a futásidő mérése **külön bontásban** történjen (H2D, kernel, D2H)
- a kernel futás **profilozható** legyen OpenCL event alapú időméréssel
- a teljesítmény tuningolható legyen:
  - work-group méretek (`local_size`)
  - local memory használat
- a helyesség ellenőrizhető legyen (pl. kis rács esetén CPU referencia vagy ismert minták)

---

## 3. Tervezett architektúra

A projekt két nagy részre fog bomlani:

### 3.1 Host oldal (C / OpenCL API)

A host program felelőssége várhatóan:

- OpenCL platform és eszköz kiválasztása (preferáltan GPU, fallback CPU)
- context + queue létrehozása (profilozás bekapcsolva)
- kernel(ek) betöltése és fordítása (forrásból)
- bufferek létrehozása (kétpufferes „ping-pong” sémával)
- iterációk futtatása (kernel indítás loopban)
- időmérések gyűjtése, kiírása és/vagy CSV mentése

### 3.2 Device oldal (OpenCL kernel)

Tervezetten legalább két kernelváltozat készül:

1) **Naiv kernel**
- minden cella a globális memóriából olvassa a szomszédokat
- egyszerű, könnyen ellenőrizhető baseline

2) **Tiled / local memory kernel (optimalizált)**
- work-group szinten a rács egy „csempéje” betöltődik **local memóriába**
- a szomszédszámítás így kevesebb globális memóriaeléréssel történik
- a local memória mérete **dinamikusan** átadható (`__local` argumentum)
- a work-group méret a host oldalról paraméterezhető (tuning)

---

## 4. Tervezett paraméterezés (CLI)

A futtatható program várhatóan parancssori argumentumokkal lesz vezérelhető, például:

- `--rows <N>` / `--cols <M>` – rácsméret
- `--iters <K>` – iterációk száma
- `--seed <S>` – random kezdőállapot seed
- `--wrap 0|1` – határfeltétel
- `--tiled 0|1` – kernelválasztás
- `--lx <X>` / `--ly <Y>` – local work-group méret
- `--csv <path>` – mérési eredmények mentése

---

## 5. Tervezett mérési terv (benchmark)

A beadandó értéke ott fog igazán kijönni, ahol nem csak „fut”, hanem **mérhetően gyorsabb** és megmutatható, hogy *miért*.

A mérési terv várható elemei:

### 5.1 Problémaméret skálázás
- 512×512, 1024×1024, 2048×2048 (és igény szerint nagyobb)
- naiv vs tiled összehasonlítás

### 5.2 Work-group tuning
- fix grid mellett több `local_size` kombináció tesztelése:
  - 8×8, 16×16, 32×8, stb.
- cél: megtalálni a hardverhez „legjobban passzoló” méretet

### 5.3 Határfeltétel hatása
- `wrap=0` vs `wrap=1`
- különösen tiled esetben érdekes a halo-kezelés miatt

### 5.4 Mérés bontása
Az eredmények várhatóan külön sorokon/mezőkben szerepelnek:

- **H2D** (host → device másolás ideje)
- **Kernel** (összes kernelidő / átlag kernelidő)
- **D2H** (device → host)
- **Total** (teljes futásidő)

A mérési adatokat célszerű lesz CSV-be menteni, hogy grafikon készülhessen (Excel/LibreOffice/plot).

---

## 6. Tervezett könyvtárstruktúra

A projekt várható felépítése:

```
gol_opencl/
  main.c
  Makefile
  include/
    kernel_loader.h
  src/
    kernel_loader.c
  kernels/
    gol_naive.cl
    gol_tiled.cl
```

---

## 7. Fordítás és futtatás (tervezett)

### 7.1 Előfeltételek
- C fordító (GCC / clang / MSVC)
- OpenCL fejlécek + ICD loader + megfelelő driver (NVIDIA/AMD/Intel)

### 7.2 Build
A gyökérmappában:

```bash
make
```

### 7.3 Futtatás (példák)
```bash
./gol_opencl --rows 2048 --cols 2048 --iters 500 --tiled 1 --lx 16 --ly 16 --wrap 1 --csv results.csv
```

---

## 8. Mitől lesz „jó” ez a beadandó?

A projekt akkor lesz erős, ha az alábbiak teljesülnek:

- van **baseline** (naiv kernel), amihez képest mérünk
- van **optimalizált** változat (local memory + tiled)
- van **tuning** (local size paraméterezhető, és a mérésekben látszik a hatása)
- van **mérési jegyzőkönyv** (CSV + grafikonok)
- a működés **ellenőrzött** (helyességi teszt vagy ismert minták)

---

## License

A beadandóhoz (ha nem írja elő másképp a tárgy) egyszerűen választható MIT licenc, vagy a tantárgyi előírások szerinti licenc/megjelölés.

