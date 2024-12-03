# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native

# Source files
SRCS = benchmark.c
OBJS = $(SRCS:.c=.o)
TARGET = ahb_sort

# Debug build settings
DEBUG_DIR = debug
DEBUG_TARGET = $(DEBUG_DIR)/$(TARGET)
DEBUG_CFLAGS = -g -DDEBUG

# Release build settings
RELEASE_DIR = release
RELEASE_TARGET = $(RELEASE_DIR)/$(TARGET)

# Default target
all: release

# Debug build
debug: CFLAGS += $(DEBUG_CFLAGS)
debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SRCS)
	@mkdir -p $(DEBUG_DIR)
	$(CC) $(CFLAGS) $^ -o $@

# Release build
release: $(RELEASE_TARGET)

$(RELEASE_TARGET): $(SRCS)
	@mkdir -p $(RELEASE_DIR)
	$(CC) $(CFLAGS) $^ -o $@

# Clean build files
clean:
	rm -rf $(DEBUG_DIR) $(RELEASE_DIR) $(OBJS)

# Run benchmarks
benchmark: release
	./$(RELEASE_TARGET)

# Run debug version
run_debug: debug
	./$(DEBUG_TARGET)

# Run release version
run: release
	./$(RELEASE_TARGET)

.PHONY: all debug release clean benchmark run_debug run
