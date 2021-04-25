                .text
                .section .text.startup,"ax"    
                .globl Reset_Handler
Reset_Handler:
                LDR sp, =__StackTop
                BL main
                B .
