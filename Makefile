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

# Comparison against stronger baselines: std::sort, pdqsort, ska_sort and two
# radix sorts. pdqsort.h and ska_sort.hpp are downloaded on first use and are
# not part of this repository.
CXX ?= c++
BASELINES_TARGET = $(RELEASE_DIR)/baselines
THIRD_PARTY = experiments/third_party

$(THIRD_PARTY)/pdqsort.h:
	@mkdir -p $(THIRD_PARTY)
	curl -sfL -o $@ https://raw.githubusercontent.com/orlp/pdqsort/master/pdqsort.h

$(THIRD_PARTY)/ska_sort.hpp:
	@mkdir -p $(THIRD_PARTY)
	curl -sfL -o $@ https://raw.githubusercontent.com/skarupke/ska_sort/master/ska_sort.hpp

$(BASELINES_TARGET): experiments/baselines.cpp ahb_sort.h $(THIRD_PARTY)/pdqsort.h $(THIRD_PARTY)/ska_sort.hpp
	@mkdir -p $(RELEASE_DIR)
	$(CXX) -std=c++17 -Wall -Wextra -Wno-deprecated-declarations -O3 -march=native -mtune=native $< -o $@

baselines: $(BASELINES_TARGET)
	./$(BASELINES_TARGET)

# Run debug version
run_debug: debug
	./$(DEBUG_TARGET)

# Run release version
run: release
	./$(RELEASE_TARGET)

.PHONY: all debug release clean benchmark test variants baselines run_debug run
