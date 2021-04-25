/**
 * \file runtime/example/Nonstandard_OS/tee/ta/include/megcc_inference_ta.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#ifndef TA_MEGCC_INFERENCE_H
#define TA_MEGCC_INFERENCE_H

/*
 * This UUID is generated with uuidgen
 * the ITU-T UUID generator at http://www.itu.int/ITU-T/asn1/uuid.html
 */
#define TA_MEGCC_INFERENCE_UUID                            \
    {                                                      \
        0x8715dc00, 0x8a5e, 0x11ec, {                      \
            0xa8, 0xa3, 0x02, 0x42, 0xac, 0x12, 0x00, 0x02 \
        }                                                  \
    }

/* The function IDs implemented in this TA */
#define TA_MEGCC_INFERENCE_CMD_INIT_MODEL 0
#define TA_MEGCC_INFERENCE_CMD_RUN_MODEL 1
#define TA_MEGCC_INFERENCE_CMD_FREE_MODEL 2

#endif /*TA_MEGCC_INFERENCE_H*/
