# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -mtune=native

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

$(DEBUG_TARGET): $(SRCS) ahb_sort.h
	@mkdir -p $(DEBUG_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $@ -lm

# Release build
release: $(RELEASE_TARGET)

$(RELEASE_TARGET): $(SRCS) ahb_sort.h
	@mkdir -p $(RELEASE_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $@ -lm

# Clean build files
clean:
	rm -rf $(DEBUG_DIR) $(RELEASE_DIR) $(OBJS)

# Run benchmarks
benchmark: release
	./$(RELEASE_TARGET)

# Correctness tests, built with sanitizers
TEST_TARGET = $(DEBUG_DIR)/test_ahb_sort
TEST_CFLAGS = -Wall -Wextra -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined

$(TEST_TARGET): test_ahb_sort.c ahb_sort.h
	@mkdir -p $(DEBUG_DIR)
	$(CC) $(TEST_CFLAGS) $< -o $@ -lm

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Timing comparison of experimental variants (see experiments/)
VARIANT_TARGET = $(RELEASE_DIR)/variant_bench

$(VARIANT_TARGET): experiments/variant_bench.c experiments/ahb_variant.inc ahb_sort.h
	@mkdir -p $(RELEASE_DIR)
	$(CC) $(CFLAGS) $< -o $@ -lm

variants: $(VARIANT_TARGET)
	./$(VARIANT_TARGET)

# Run debug version
run_debug: debug
	./$(DEBUG_TARGET)

# Run release version
run: release
	./$(RELEASE_TARGET)

.PHONY: all debug release clean benchmark test variants run_debug run
