# Makefile (再修正)

.DEFAULT_GOAL := all

THIRDPARTY_DIR := $(CURDIR)/thirdparty
CUCOLL_DIR     := $(THIRDPARTY_DIR)/cuCollections
CCCL_DIR       := $(THIRDPARTY_DIR)/CCCL

NVCC       := nvcc
NVCCFLAGS  := -std=c++17 --expt-extended-lambda -arch=sm_90
INCLUDES   := -I$(CUCOLL_DIR)/include \
              -I$(CCCL_DIR)/thrust \
              -I$(CCCL_DIR)/libcudacxx/include \
              -I$(CCCL_DIR)/cub

TARGET := epic
SRC    := main.cu

.PHONY: init all clean

# ---- 依存ライブラリ取得 ----
init:
	@mkdir -p $(THIRDPARTY_DIR)
	@if [ ! -d "$(CUCOLL_DIR)" ]; then \
		echo "Cloning cuCollections ..."; \
		cd $(THIRDPARTY_DIR) && git clone https://github.com/NVIDIA/cuCollections.git; \
	else \
		echo "cuCollections already present — skip clone"; \
	fi
	@if [ ! -d "$(CCCL_DIR)" ]; then \
		echo "Cloning CCCL ..."; \
		cd $(THIRDPARTY_DIR) && git clone https://github.com/NVIDIA/cccl.git CCCL; \
	else \
		echo "CCCL already present — skip clone"; \
	fi

# ---- ビルド ----
all: $(TARGET)

# 依存が無ければ自動で init
$(TARGET): init $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $@
	@echo "Built $(TARGET)"

# ---- クリーン ----
clean:
	@rm -f $(TARGET)
	@echo "Cleaned"
