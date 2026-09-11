# Unified Makefile for AGC_STFT_GRU
#
# 這個 Makefile 放在 src/ 資料夾的「同一層」(專案根目錄)。
# 兩個 build 目標都是從 src/*.c 編譯：
#   1. libs/libnnagc.so.1  - 共享函式庫，用來產生訓練用的特徵
#                             (排除 demo_agc.c，因為它含 main()，
#                              不該被編進共享函式庫)
#   2. ./demo_agc           - 獨立的 demo 執行檔，處理 wav 檔
#                              (輸入 wav -> AGC 處理 -> 輸出 wav)
#
# 目錄結構假設：
#   ./Makefile
#   ./src/*.c, ./src/*.h        <- 所有原始碼與標頭檔都在這裡
#   ./vad/include/...           <- (選用) 額外的 include 路徑
#
# Usage:
#   make            # 同時建置 libs/libnnagc.so.1 與 ./demo_agc
#   make lib        # 只建置共享函式庫
#   make demo       # 只建置 demo 執行檔
#   make run IN=input.wav OUT=output_agc.wav
#   make clean

CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -O3 -std=c99 -fPIC -Isrc -Ivad/include
LDFLAGS ?= -lm

SRC_DIR := src
OBJ_DIR := build
LIB_DIR := libs

# src/ 底下所有的 .c / .h
SRCS    := $(wildcard $(SRC_DIR)/*.c)
OBJS    := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
HEADERS := $(wildcard $(SRC_DIR)/*.h)

# --- 共享函式庫 (libnnagc.so.1) ------------------------------------------
# demo_agc.c 含 main()，不能一起連進共享函式庫，故過濾掉。
LIB_TARGET := $(LIB_DIR)/libnnagc.so.1
LIB_OBJS   := $(filter-out $(OBJ_DIR)/demo_agc.o,$(OBJS))

# --- Demo 執行檔 (wav 進 -> wav 出) ---------------------------------------
DEMO_TARGET := demo_agc
DEMO_OBJS   := $(OBJS)

IN  ?= input.wav
OUT ?= output_agc.wav

.PHONY: all lib demo run clean

all: lib demo

lib: $(LIB_TARGET)

demo: $(DEMO_TARGET)

$(LIB_TARGET): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

$(DEMO_TARGET): $(DEMO_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(DEMO_TARGET)
	./$(DEMO_TARGET) $(IN) $(OUT)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(DEMO_TARGET)
