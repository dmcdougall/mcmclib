#CC=gcc
CFLAGS=-fno-common -fPIC -I/opt/local/include
LDFLAGS=-L/opt/local/lib -lm -lgsl -lfftw3 -framework Accelerate
LIB_NAME=libmcmc
CURRENT_VERSION=1.0.0
COMPATIBILITY_VERSION=1.0
MAJOR_VERSION=1
INSTALL_DIR=/Users/damon/lib
HEADER_DIR=/Users/damon/include
INSTALL_NAME=$(INSTALL_DIR)/$(LIB_NAME).$(MAJOR_VERSION).dylib

# Rules
###################################
%.o : %.c
	$(CC) -c $< $(CFLAGS) -o $@
###################################

OBJS = infmcmc.o finmcmc.o

HEADERS = infmcmc.h finmcmc.h

all: mcmclib

#infmcmc.o: infmcmc.c
#	$(CC) $(CFLAGS) $(.TARGET) -c $<

#finmcmc.o: finmcmc.c
#	$(CC) $(CFLAGS) $(.TARGET) -c $<

mcmclib: $(OBJS) $(HEADERS)
	$(CC) -dynamiclib -install_name $(INSTALL_NAME) \
	-compatibility_version $(COMPATIBILITY_VERSION) \
	-current_version $(CURRENT_VERSION) \
	$(LDFLAGS) -o $(LIB_NAME).$(CURRENT_VERSION).dylib $(OBJS)

install:
	cp *.h $(HEADER_DIR)
	cp $(LIB_NAME).$(CURRENT_VERSION).dylib $(INSTALL_DIR)/$(LIB_NAME).$(CURRENT_VERSION).dylib
	ln -s $(INSTALL_DIR)/$(LIB_NAME).$(CURRENT_VERSION).dylib $(INSTALL_DIR)/$(LIB_NAME).$(MAJOR_VERSION).dylib
	ln -s $(INSTALL_DIR)/$(LIB_NAME).$(CURRENT_VERSION).dylib $(INSTALL_DIR)/$(LIB_NAME).dylib

clean:
	rm infmcmc.o
	rm finmcmc.o
	rm $(LIB_NAME).$(CURRENT_VERSION).dylib
	rm *~

uninstall:
	rm $(INSTALL_DIR)/libmcmc.*
