CXX_FLAGS=-Wall -Wextra -Wpedantic -Wshadow -Werror
DEBUG_FLAGS=-O0 -g# -fsanitize=address,undefined,leak,float-divide-by-zero
OPT_FLAGS=-O3 -DNDEBUG -march=native -mtune=native

debug: main.cpp
	g++ --std=c++17 $(DEBUG_FLAGS) $(CXX_FLAGS) main.cpp -o main.x

release: main.cpp
	g++ --std=c++17 $(OPT_FLAGS) $(CXX_FLAGS) main.cpp -o main.x

format:
	clang-format --style=file -i *.cpp *.h

clean:
	rm -f main.x
