#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

/* To get the TA UUID definition */
#include <megcc_inference_ta.h>

#define TA_UUID TA_MEGCC_INFERENCE_UUID

/*
 * TA properties: multi-instance TA, no specific attribute
 * TA_FLAG_EXEC_DDR is meaningless but mandated.
 */
#define TA_FLAGS TA_FLAG_EXEC_DDR

/* Provisioned stack size */
#define TA_STACK_SIZE (64 * 1024)

/* Provisioned heap size for TEE_Malloc() and friends */
#define TA_DATA_SIZE (4096 * 1024)

/* The gpd.ta.version property */
#define TA_VERSION "1.0"

/* The gpd.ta.description property */
#define TA_DESCRIPTION "Example of OP-TEE MegCC inference World Trusted Application"

#endif /* USER_TA_HEADER_DEFINES_H */
