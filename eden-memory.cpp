#include "eden-memory.h"

eden_memory_status eden_memory_status_combine(eden_memory_status s0, eden_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case EDEN_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case EDEN_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case EDEN_MEMORY_STATUS_FAILED_PREPARE:
        case EDEN_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case EDEN_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case EDEN_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case EDEN_MEMORY_STATUS_FAILED_PREPARE:
        case EDEN_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? EDEN_MEMORY_STATUS_SUCCESS : EDEN_MEMORY_STATUS_NO_UPDATE;
}

bool eden_memory_status_is_fail(eden_memory_status status) {
    switch (status) {
        case EDEN_MEMORY_STATUS_SUCCESS:
        case EDEN_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case EDEN_MEMORY_STATUS_FAILED_PREPARE:
        case EDEN_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
