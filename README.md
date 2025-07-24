# Epic[osdi'24]の再実装及び提案手法の実装

## Epicの実装について
WIP

## 環境構築
1. `make init`を実行
2. `cuCollections/include/cuco/detail/__config`の33~35行目（以下の部分）をコメントアウト
```
#if !defined(CCCL_VERSION) || (CCCL_VERSION < 3000000)
#error "CCCL version 3.0.0 or later is required"
#endif
```
3. `make`を実行
`epic`としてコンパイル