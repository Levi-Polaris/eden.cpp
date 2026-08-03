#pragma once

int eden_build_number(void);

const char * eden_commit(void);
const char * eden_compiler(void);

const char * eden_build_target(void);
const char * eden_build_info(void);

void eden_print_build_info(void);
