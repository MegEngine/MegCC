                .title startup64.s
                .arch armv8-a
                .text
                .section .text.startup,"ax"    
                .globl Reset_Handler
Reset_Handler:
                ldr x0, =__StackTop
                mov sp, x0
                bl  main
wait:           wfe
                b wait
               .end