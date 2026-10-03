
/home/avifenesh/.local/state/memra-rig-darklanes-20261002/receipts/B/atomic-update-978/emit-before-server/server-tests-before:	file format elf64-x86-64

Disassembly of section .text:

000000000352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>:
 352d6c0:      	cmpl	$-0x1, (%rdi)
 352d6c3:      	jne	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 352d6c9:      	addq	$0x8, %rdi
 352d6cd:      	jmp	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 352d6d2:      	int3
 352d6d3:      	int3
 352d6d4:      	int3
 352d6d5:      	int3
 352d6d6:      	int3
 352d6d7:      	int3
 352d6d8:      	int3
 352d6d9:      	int3
 352d6da:      	int3
 352d6db:      	int3
 352d6dc:      	int3
 352d6dd:      	int3
 352d6de:      	int3
 352d6df:      	int3

000000000352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>:
 352d8b0:      	pushq	%r15
 352d8b2:      	pushq	%r14
 352d8b4:      	pushq	%rbx
 352d8b5:      	movq	%rdi, %rbx
 352d8b8:      	callq	0x3550c90 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs1S6izGDSMYD_4http6header3map9HeaderMapECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 352d8bd:      	movq	0x60(%rbx), %r15
 352d8c1:      	testq	%r15, %r15
 352d8c4:      	je	0x352d8e1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x31>
 352d8c6:      	movq	%r15, %rdi
 352d8c9:      	callq	0x392c3a0 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCs4NRVxsYgnAr_4core3any6TypeIdINtNtCscdodAO9FK5_5alloc5boxed3BoxDNtNtCs1S6izGDSMYD_4http10extensions8AnyCloneNtNtBT_6marker4SendNtB2G_4SyncEL_EEENtNtNtBT_3ops4drop4Drop4dropCs3pwlnhBXFtN_12memra_server>
 352d8ce:      	movl	$0x20, %esi
 352d8d3:      	movl	$0x8, %edx
 352d8d8:      	movq	%r15, %rdi
 352d8db:      	callq	*0x1c1345f(%rip)        # 0x5140d40 <writev+0x5140d40>
 352d8e1:      	movq	0x70(%rbx), %r15
 352d8e5:      	movq	0x78(%rbx), %rbx
 352d8e9:      	movq	(%rbx), %rax
 352d8ec:      	testq	%rax, %rax
 352d8ef:      	je	0x352d8f6 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x46>
 352d8f1:      	movq	%r15, %rdi
 352d8f4:      	callq	*%rax
 352d8f6:      	movq	0x8(%rbx), %rsi
 352d8fa:      	testq	%rsi, %rsi
 352d8fd:      	je	0x352d911 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x61>
 352d8ff:      	movq	0x10(%rbx), %rdx
 352d903:      	movq	%r15, %rdi
 352d906:      	popq	%rbx
 352d907:      	popq	%r14
 352d909:      	popq	%r15
 352d90b:      	jmpq	*0x1c1342f(%rip)        # 0x5140d40 <writev+0x5140d40>
 352d911:      	popq	%rbx
 352d912:      	popq	%r14
 352d914:      	popq	%r15
 352d916:      	retq
 352d917:      	movq	%rax, %r14
 352d91a:      	movq	0x8(%rbx), %rsi
 352d91e:      	testq	%rsi, %rsi
 352d921:      	je	0x352d969 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0xb9>
 352d923:      	movq	0x10(%rbx), %rdx
 352d927:      	movq	%r15, %rdi
 352d92a:      	callq	*0x1c13410(%rip)        # 0x5140d40 <writev+0x5140d40>
 352d930:      	movq	%r14, %rdi
 352d933:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 352d938:      	movq	%rax, %r14
 352d93b:      	movl	$0x20, %esi
 352d940:      	movl	$0x8, %edx
 352d945:      	movq	%r15, %rdi
 352d948:      	callq	*0x1c133f2(%rip)        # 0x5140d40 <writev+0x5140d40>
 352d94e:      	jmp	0x352d95c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0xac>
 352d950:      	movq	%rax, %r14
 352d953:      	movq	0x60(%rbx), %rdi
 352d957:      	callq	0x354d360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs1S6izGDSMYD_4http10extensions10ExtensionsECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 352d95c:      	movq	0x70(%rbx), %rdi
 352d960:      	movq	0x78(%rbx), %rsi
 352d964:      	callq	0x352fd30 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCsaSG9NyffgI5_9axum_core4body4BodyECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 352d969:      	movq	%r14, %rdi
 352d96c:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 352d971:      	callq	*0x1c13441(%rip)        # 0x5140db8 <writev+0x5140db8>
 352d977:      	callq	*0x1c1343b(%rip)        # 0x5140db8 <writev+0x5140db8>
 352d97d:      	int3
 352d97e:      	int3
 352d97f:      	int3

000000000354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 354c0c0:      	pushq	%r14
 354c0c2:      	pushq	%rbx
 354c0c3:      	pushq	%rax
 354c0c4:      	movq	%rdi, %rbx
 354c0c7:      	cmpb	$0x1, 0x31(%rdi)
 354c0cb:      	jne	0x354c121 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 354c0cd:      	movq	0x8(%rbx), %rcx
 354c0d1:      	movq	(%rcx), %rax
 354c0d4:      	nopw	%cs:(%rax,%rax)
 354c0e0:      	testq	%rax, %rax
 354c0e3:      	je	0x354c0f0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 354c0e5:      	leaq	-0x1(%rax), %rdx
 354c0e9:      	lock
 354c0ea:      	cmpxchgq	%rdx, (%rcx)
 354c0ee:      	jne	0x354c0e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 354c0f0:      	cmpb	$0x0, 0x32(%rbx)
 354c0f4:      	jne	0x354c121 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 354c0f6:      	movq	(%rbx), %rcx
 354c0f9:      	movzbl	0x30(%rbx), %edx
 354c0fd:      	movq	(%rcx,%rdx,8), %rax
 354c101:      	nopw	%cs:(%rax,%rax)
 354c110:      	testq	%rax, %rax
 354c113:      	je	0x354c121 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 354c115:      	leaq	-0x1(%rax), %rsi
 354c119:      	lock
 354c11a:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 354c11f:      	jne	0x354c110 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 354c121:      	cmpl	$-0x1, 0x28(%rbx)
 354c125:      	je	0x354c16a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 354c127:      	movq	0x10(%rbx), %rdi
 354c12b:      	cmpq	$0x2, %rdi
 354c12f:      	ja	0x354c172 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 354c131:      	movq	0x18(%rbx), %rcx
 354c135:      	addq	$0x18, %rbx
 354c139:      	movq	0x30(%rcx,%rdi,8), %rax
 354c13e:      	nop
 354c140:      	testq	%rax, %rax
 354c143:      	je	0x354c152 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 354c145:      	leaq	-0x1(%rax), %rdx
 354c149:      	lock
 354c14a:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 354c150:      	jne	0x354c140 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 354c152:      	movq	(%rbx), %rax
 354c155:      	lock
 354c156:      	decq	(%rax)
 354c159:      	jne	0x354c16a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 354c15b:      	movq	%rbx, %rdi
 354c15e:      	addq	$0x8, %rsp
 354c162:      	popq	%rbx
 354c163:      	popq	%r14
 354c165:      	jmp	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 354c16a:      	addq	$0x8, %rsp
 354c16e:      	popq	%rbx
 354c16f:      	popq	%r14
 354c171:      	retq
 354c172:      	leaq	0x1ad5d57(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 354c179:      	movl	$0x3, %esi
 354c17e:      	callq	*0x1bf4cf4(%rip)        # 0x5140e78 <writev+0x5140e78>
 354c184:      	ud2
 354c186:      	movq	%rax, %r14
 354c189:      	movq	0x18(%rbx), %rax
 354c18d:      	lock
 354c18e:      	decq	(%rax)
 354c191:      	jne	0x354c19f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 354c193:      	addq	$0x18, %rbx
 354c197:      	movq	%rbx, %rdi
 354c19a:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 354c19f:      	movq	%r14, %rdi
 354c1a2:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 354c1a7:      	callq	*0x1bf4c0b(%rip)        # 0x5140db8 <writev+0x5140db8>
 354c1ad:      	int3
 354c1ae:      	int3
 354c1af:      	int3

000000000354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475>:
 354c4e0:      	pushq	%r14
 354c4e2:      	pushq	%rbx
 354c4e3:      	pushq	%rax
 354c4e4:      	movq	%rdi, %rbx
 354c4e7:      	callq	0x3c723a0 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 354c4ec:      	movq	0x28(%rbx), %rax
 354c4f0:      	lock
 354c4f1:      	decq	(%rax)
 354c4f4:      	jne	0x354c4ff <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x1f>
 354c4f6:      	leaq	0x28(%rbx), %rdi
 354c4fa:      	callq	0x39ba040 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtB7_3vec3VecNtNtB7_6string6StringEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c4ff:      	movq	0x30(%rbx), %rax
 354c503:      	lock
 354c504:      	decq	(%rax)
 354c507:      	jne	0x354c512 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x32>
 354c509:      	leaq	0x30(%rbx), %rdi
 354c50d:      	callq	0x39bb0e0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEE9drop_slowB1Z_>
 354c512:      	movq	0x38(%rbx), %rax
 354c516:      	lock
 354c517:      	decq	(%rax)
 354c51a:      	jne	0x354c525 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x45>
 354c51c:      	leaq	0x38(%rbx), %rdi
 354c520:      	callq	0x39bc090 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockIBw_NtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEE9drop_slowB1C_>
 354c525:      	movq	0x80(%rbx), %rax
 354c52c:      	testq	%rax, %rax
 354c52f:      	je	0x354c543 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x63>
 354c531:      	lock
 354c532:      	decq	(%rax)
 354c535:      	jne	0x354c543 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x63>
 354c537:      	leaq	0x80(%rbx), %rdi
 354c53e:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c543:      	movq	0x90(%rbx), %rax
 354c54a:      	testq	%rax, %rax
 354c54d:      	je	0x354c561 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x81>
 354c54f:      	lock
 354c550:      	decq	(%rax)
 354c553:      	jne	0x354c561 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x81>
 354c555:      	leaq	0x90(%rbx), %rdi
 354c55c:      	callq	0x39bb070 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringIBw_NtCscIVK9LJ3de3_15memra_tokenizer9TokenizerEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c561:      	movq	0x98(%rbx), %rax
 354c568:      	testq	%rax, %rax
 354c56b:      	je	0x354c580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xa0>
 354c56d:      	lock
 354c56e:      	decq	(%rax)
 354c571:      	jne	0x354c580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xa0>
 354c573:      	leaq	0x98(%rbx), %rdi
 354c57a:      	callq	*0x1bf5a28(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c580:      	movq	0x10(%rbx), %rax
 354c584:      	testq	%rax, %rax
 354c587:      	je	0x354c599 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xb9>
 354c589:      	lock
 354c58a:      	decq	(%rax)
 354c58d:      	jne	0x354c599 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xb9>
 354c58f:      	leaq	0x10(%rbx), %rdi
 354c593:      	callq	*0x1bf5a0f(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c599:      	movq	0x40(%rbx), %rax
 354c59d:      	lock
 354c59e:      	decq	(%rax)
 354c5a1:      	jne	0x354c5ac <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xcc>
 354c5a3:      	leaq	0x40(%rbx), %rdi
 354c5a7:      	callq	0x39bb8c0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEE9drop_slowB1y_>
 354c5ac:      	movq	0x48(%rbx), %rax
 354c5b0:      	lock
 354c5b1:      	decq	(%rax)
 354c5b4:      	jne	0x354c5bf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xdf>
 354c5b6:      	leaq	0x48(%rbx), %rdi
 354c5ba:      	callq	0x39b9f80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6worker15EventQueueStateE9drop_slowBJ_>
 354c5bf:      	movq	0x50(%rbx), %rax
 354c5c3:      	lock
 354c5c4:      	decq	(%rax)
 354c5c7:      	jne	0x354c5d2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0xf2>
 354c5c9:      	leaq	0x50(%rbx), %rdi
 354c5cd:      	callq	0x39bb4f0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringjEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c5d2:      	movq	0x58(%rbx), %rax
 354c5d6:      	lock
 354c5d7:      	decq	(%rax)
 354c5da:      	jne	0x354c5e5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x105>
 354c5dc:      	leaq	0x58(%rbx), %rdi
 354c5e0:      	callq	0x39bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 354c5e5:      	movq	0xb0(%rbx), %rax
 354c5ec:      	testq	%rax, %rax
 354c5ef:      	je	0x354c603 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x123>
 354c5f1:      	lock
 354c5f2:      	decq	(%rax)
 354c5f5:      	jne	0x354c603 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x123>
 354c5f7:      	leaq	0xb0(%rbx), %rdi
 354c5fe:      	callq	0x39bd080 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server8darklane10BgJobStateE9drop_slowBJ_>
 354c603:      	movq	0x60(%rbx), %rax
 354c607:      	lock
 354c608:      	decq	(%rax)
 354c60b:      	jne	0x354c616 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x136>
 354c60d:      	leaq	0x60(%rbx), %rdi
 354c611:      	callq	0x39bbe80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCsdMwdNqnNPrU_11memra_lanes12audio_stream14AudioSchedulerEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c616:      	movq	0x68(%rbx), %rax
 354c61a:      	lock
 354c61b:      	decq	(%rax)
 354c61e:      	jne	0x354c629 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x149>
 354c620:      	leaq	0x68(%rbx), %rdi
 354c624:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c629:      	movq	0x78(%rbx), %rax
 354c62d:      	lock
 354c62e:      	decq	(%rax)
 354c631:      	jne	0x354c646 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x166>
 354c633:      	addq	$0x78, %rbx
 354c637:      	movq	%rbx, %rdi
 354c63a:      	addq	$0x8, %rsp
 354c63e:      	popq	%rbx
 354c63f:      	popq	%r14
 354c641:      	jmp	0x39bb480 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server15background_jobs7ControlEEE9drop_slowB2z_>
 354c646:      	addq	$0x8, %rsp
 354c64a:      	popq	%rbx
 354c64b:      	popq	%r14
 354c64d:      	retq
 354c64e:      	movq	%rax, %r14
 354c651:      	jmp	0x354c7cf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2ef>
 354c656:      	movq	%rax, %r14
 354c659:      	jmp	0x354c765 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x285>
 354c65e:      	movq	%rax, %r14
 354c661:      	jmp	0x354c74c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x26c>
 354c666:      	movq	%rax, %r14
 354c669:      	jmp	0x354c72d <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x24d>
 354c66e:      	movq	%rax, %r14
 354c671:      	jmp	0x354c70f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x22f>
 354c676:      	movq	%rax, %r14
 354c679:      	jmp	0x354c7f5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x315>
 354c67e:      	movq	%rax, %r14
 354c681:      	jmp	0x354c7e2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x302>
 354c686:      	movq	%rax, %r14
 354c689:      	jmp	0x354c7b1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2d1>
 354c68e:      	movq	%rax, %r14
 354c691:      	jmp	0x354c79e <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2be>
 354c696:      	movq	%rax, %r14
 354c699:      	jmp	0x354c78b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2ab>
 354c69e:      	movq	%rax, %r14
 354c6a1:      	jmp	0x354c778 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x298>
 354c6a6:      	movq	%rax, %r14
 354c6a9:      	jmp	0x354c6f1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x211>
 354c6ab:      	movq	%rax, %r14
 354c6ae:      	jmp	0x354c6de <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x1fe>
 354c6b0:      	movq	%rax, %r14
 354c6b3:      	jmp	0x354c6cb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x1eb>
 354c6b5:      	movq	%rax, %r14
 354c6b8:      	movq	0x28(%rbx), %rax
 354c6bc:      	lock
 354c6bd:      	decq	(%rax)
 354c6c0:      	jne	0x354c6cb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x1eb>
 354c6c2:      	leaq	0x28(%rbx), %rdi
 354c6c6:      	callq	0x39ba040 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtB7_3vec3VecNtNtB7_6string6StringEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c6cb:      	movq	0x30(%rbx), %rax
 354c6cf:      	lock
 354c6d0:      	decq	(%rax)
 354c6d3:      	jne	0x354c6de <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x1fe>
 354c6d5:      	leaq	0x30(%rbx), %rdi
 354c6d9:      	callq	0x39bb0e0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEE9drop_slowB1Z_>
 354c6de:      	movq	0x38(%rbx), %rax
 354c6e2:      	lock
 354c6e3:      	decq	(%rax)
 354c6e6:      	jne	0x354c6f1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x211>
 354c6e8:      	leaq	0x38(%rbx), %rdi
 354c6ec:      	callq	0x39bc090 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockIBw_NtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEE9drop_slowB1C_>
 354c6f1:      	movq	0x80(%rbx), %rax
 354c6f8:      	testq	%rax, %rax
 354c6fb:      	je	0x354c70f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x22f>
 354c6fd:      	lock
 354c6fe:      	decq	(%rax)
 354c701:      	jne	0x354c70f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x22f>
 354c703:      	leaq	0x80(%rbx), %rdi
 354c70a:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c70f:      	movq	0x90(%rbx), %rax
 354c716:      	testq	%rax, %rax
 354c719:      	je	0x354c72d <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x24d>
 354c71b:      	lock
 354c71c:      	decq	(%rax)
 354c71f:      	jne	0x354c72d <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x24d>
 354c721:      	leaq	0x90(%rbx), %rdi
 354c728:      	callq	0x39bb070 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringIBw_NtCscIVK9LJ3de3_15memra_tokenizer9TokenizerEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c72d:      	movq	0x98(%rbx), %rax
 354c734:      	testq	%rax, %rax
 354c737:      	je	0x354c74c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x26c>
 354c739:      	lock
 354c73a:      	decq	(%rax)
 354c73d:      	jne	0x354c74c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x26c>
 354c73f:      	leaq	0x98(%rbx), %rdi
 354c746:      	callq	*0x1bf585c(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c74c:      	movq	0x10(%rbx), %rax
 354c750:      	testq	%rax, %rax
 354c753:      	je	0x354c765 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x285>
 354c755:      	lock
 354c756:      	decq	(%rax)
 354c759:      	jne	0x354c765 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x285>
 354c75b:      	leaq	0x10(%rbx), %rdi
 354c75f:      	callq	*0x1bf5843(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c765:      	movq	0x40(%rbx), %rax
 354c769:      	lock
 354c76a:      	decq	(%rax)
 354c76d:      	jne	0x354c778 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x298>
 354c76f:      	leaq	0x40(%rbx), %rdi
 354c773:      	callq	0x39bb8c0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEE9drop_slowB1y_>
 354c778:      	movq	0x48(%rbx), %rax
 354c77c:      	lock
 354c77d:      	decq	(%rax)
 354c780:      	jne	0x354c78b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2ab>
 354c782:      	leaq	0x48(%rbx), %rdi
 354c786:      	callq	0x39b9f80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6worker15EventQueueStateE9drop_slowBJ_>
 354c78b:      	movq	0x50(%rbx), %rax
 354c78f:      	lock
 354c790:      	decq	(%rax)
 354c793:      	jne	0x354c79e <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2be>
 354c795:      	leaq	0x50(%rbx), %rdi
 354c799:      	callq	0x39bb4f0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringjEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c79e:      	movq	0x58(%rbx), %rax
 354c7a2:      	lock
 354c7a3:      	decq	(%rax)
 354c7a6:      	jne	0x354c7b1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2d1>
 354c7a8:      	leaq	0x58(%rbx), %rdi
 354c7ac:      	callq	0x39bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 354c7b1:      	movq	0xb0(%rbx), %rax
 354c7b8:      	testq	%rax, %rax
 354c7bb:      	je	0x354c7cf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2ef>
 354c7bd:      	lock
 354c7be:      	decq	(%rax)
 354c7c1:      	jne	0x354c7cf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x2ef>
 354c7c3:      	leaq	0xb0(%rbx), %rdi
 354c7ca:      	callq	0x39bd080 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server8darklane10BgJobStateE9drop_slowBJ_>
 354c7cf:      	movq	0x60(%rbx), %rax
 354c7d3:      	lock
 354c7d4:      	decq	(%rax)
 354c7d7:      	jne	0x354c7e2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x302>
 354c7d9:      	leaq	0x60(%rbx), %rdi
 354c7dd:      	callq	0x39bbe80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCsdMwdNqnNPrU_11memra_lanes12audio_stream14AudioSchedulerEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c7e2:      	movq	0x68(%rbx), %rax
 354c7e6:      	lock
 354c7e7:      	decq	(%rax)
 354c7ea:      	jne	0x354c7f5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x315>
 354c7ec:      	leaq	0x68(%rbx), %rdi
 354c7f0:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c7f5:      	movq	0x78(%rbx), %rax
 354c7f9:      	lock
 354c7fa:      	decq	(%rax)
 354c7fd:      	jne	0x354c80b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475+0x32b>
 354c7ff:      	addq	$0x78, %rbx
 354c803:      	movq	%rbx, %rdi
 354c806:      	callq	0x39bb480 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server15background_jobs7ControlEEE9drop_slowB2z_>
 354c80b:      	movq	%r14, %rdi
 354c80e:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 354c813:      	callq	*0x1bf459f(%rip)        # 0x5140db8 <writev+0x5140db8>
 354c819:      	int3
 354c81a:      	int3
 354c81b:      	int3
 354c81c:      	int3
 354c81d:      	int3
 354c81e:      	int3
 354c81f:      	int3

0000000003550460 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475>:
 3550460:      	pushq	%rbp
 3550461:      	pushq	%r15
 3550463:      	pushq	%r14
 3550465:      	pushq	%r13
 3550467:      	pushq	%r12
 3550469:      	pushq	%rbx
 355046a:      	pushq	%rax
 355046b:      	movq	0x10(%rdi), %rbx
 355046f:      	testq	%rbx, %rbx
 3550472:      	je	0x355052b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0xcb>
 3550478:      	movq	%rdi, %r14
 355047b:      	movq	0x20(%rdi), %r15
 355047f:      	testq	%r15, %r15
 3550482:      	je	0x35504ee <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0x8e>
 3550484:      	movq	0x8(%r14), %r12
 3550488:      	movdqa	(%r12), %xmm0
 355048e:      	leaq	0x10(%r12), %r13
 3550493:      	pmovmskb	%xmm0, %eax
 3550497:      	notl	%eax
 3550499:      	jmp	0x35504c5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0x65>
 355049b:      	nopl	(%rax,%rax)
 35504a0:      	leal	-0x1(%rax), %ebp
 35504a3:      	tzcntl	%eax, %ecx
 35504a7:      	andl	%eax, %ebp
 35504a9:      	shll	$0x8, %ecx
 35504ac:      	movq	%r12, %rdi
 35504af:      	subq	%rcx, %rdi
 35504b2:      	addq	$-0x100, %rdi
 35504b9:      	callq	0x38c3430 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryEEB1h_.llvm.13030924631975612447>
 35504be:      	movl	%ebp, %eax
 35504c0:      	decq	%r15
 35504c3:      	je	0x35504ee <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0x8e>
 35504c5:      	testw	%ax, %ax
 35504c8:      	jne	0x35504a0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0x40>
 35504ca:      	nopw	(%rax,%rax)
 35504d0:      	movdqa	(%r13), %xmm0
 35504d6:      	addq	$-0x1000, %r12          # imm = 0xF000
 35504dd:      	addq	$0x10, %r13
 35504e1:      	pmovmskb	%xmm0, %eax
 35504e5:      	xorl	$0xffff, %eax           # imm = 0xFFFF
 35504ea:      	je	0x35504d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0x70>
 35504ec:      	jmp	0x35504a0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0x40>
 35504ee:      	movq	%rbx, %rax
 35504f1:      	shlq	$0x8, %rax
 35504f5:      	addq	%rax, %rbx
 35504f8:      	addq	$0x111, %rbx            # imm = 0x111
 35504ff:      	je	0x355052b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.14966571587344355475+0xcb>
 3550501:      	movq	0x8(%r14), %rdi
 3550505:      	subq	%rax, %rdi
 3550508:      	addq	$-0x100, %rdi
 355050f:      	movl	$0x10, %edx
 3550514:      	movq	%rbx, %rsi
 3550517:      	addq	$0x8, %rsp
 355051b:      	popq	%rbx
 355051c:      	popq	%r12
 355051e:      	popq	%r13
 3550520:      	popq	%r14
 3550522:      	popq	%r15
 3550524:      	popq	%rbp
 3550525:      	jmpq	*0x1bf0815(%rip)        # 0x5140d40 <writev+0x5140d40>
 355052b:      	addq	$0x8, %rsp
 355052f:      	popq	%rbx
 3550530:      	popq	%r12
 3550532:      	popq	%r13
 3550534:      	popq	%r14
 3550536:      	popq	%r15
 3550538:      	popq	%rbp
 3550539:      	retq
 355053a:      	int3
 355053b:      	int3
 355053c:      	int3
 355053d:      	int3
 355053e:      	int3
 355053f:      	int3

0000000003550de0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>:
 3550de0:      	pushq	%r15
 3550de2:      	pushq	%r14
 3550de4:      	pushq	%r12
 3550de6:      	pushq	%rbx
 3550de7:      	pushq	%rax
 3550de8:      	movq	(%rdi), %rax
 3550deb:      	movl	%eax, %ecx
 3550ded:      	andl	$0x3, %ecx
 3550df0:      	leal	-0x2(%rcx), %edx
 3550df3:      	cmpl	$0x2, %edx
 3550df6:      	jb	0x3550dfd <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x1d>
 3550df8:      	testq	%rcx, %rcx
 3550dfb:      	jne	0x3550e09 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x29>
 3550dfd:      	addq	$0x8, %rsp
 3550e01:      	popq	%rbx
 3550e02:      	popq	%r12
 3550e04:      	popq	%r14
 3550e06:      	popq	%r15
 3550e08:      	retq
 3550e09:      	leaq	-0x1(%rax), %rbx
 3550e0d:      	movq	-0x1(%rax), %r14
 3550e11:      	movq	0x7(%rax), %r12
 3550e15:      	movq	(%r12), %rax
 3550e19:      	testq	%rax, %rax
 3550e1c:      	je	0x3550e23 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x43>
 3550e1e:      	movq	%r14, %rdi
 3550e21:      	callq	*%rax
 3550e23:      	movq	0x8(%r12), %rsi
 3550e28:      	testq	%rsi, %rsi
 3550e2b:      	je	0x3550e3b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x5b>
 3550e2d:      	movq	0x10(%r12), %rdx
 3550e32:      	movq	%r14, %rdi
 3550e35:      	callq	*0x1beff05(%rip)        # 0x5140d40 <writev+0x5140d40>
 3550e3b:      	movl	$0x18, %esi
 3550e40:      	movl	$0x8, %edx
 3550e45:      	movq	%rbx, %rdi
 3550e48:      	addq	$0x8, %rsp
 3550e4c:      	popq	%rbx
 3550e4d:      	popq	%r12
 3550e4f:      	popq	%r14
 3550e51:      	popq	%r15
 3550e53:      	jmpq	*0x1befee7(%rip)        # 0x5140d40 <writev+0x5140d40>
 3550e59:      	movq	%rax, %r15
 3550e5c:      	movq	0x8(%r12), %rsi
 3550e61:      	testq	%rsi, %rsi
 3550e64:      	je	0x3550e74 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475+0x94>
 3550e66:      	movq	0x10(%r12), %rdx
 3550e6b:      	movq	%r14, %rdi
 3550e6e:      	callq	*0x1befecc(%rip)        # 0x5140d40 <writev+0x5140d40>
 3550e74:      	movl	$0x18, %esi
 3550e79:      	movl	$0x8, %edx
 3550e7e:      	movq	%rbx, %rdi
 3550e81:      	callq	*0x1befeb9(%rip)        # 0x5140d40 <writev+0x5140d40>
 3550e87:      	movq	%r15, %rdi
 3550e8a:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3550e8f:      	int3

0000000003552540 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEECs3pwlnhBXFtN_12memra_server>:
 3552540:      	jmp	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 3552545:      	int3
 3552546:      	int3
 3552547:      	int3
 3552548:      	int3
 3552549:      	int3
 355254a:      	int3
 355254b:      	int3
 355254c:      	int3
 355254d:      	int3
 355254e:      	int3
 355254f:      	int3

000000000366d170 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475>:
 366d170:      	pushq	%rbp
 366d171:      	movq	%rsp, %rbp
 366d174:      	pushq	%r15
 366d176:      	pushq	%r14
 366d178:      	pushq	%r13
 366d17a:      	pushq	%r12
 366d17c:      	pushq	%rbx
 366d17d:      	andq	$-0x80, %rsp
 366d181:      	subq	$0x400, %rsp            # imm = 0x400
 366d188:      	movq	%r8, 0x68(%rsp)
 366d18d:      	movl	%ecx, %ebx
 366d18f:      	movq	%rdx, %r14
 366d192:      	movq	%rsi, %r12
 366d195:      	movq	%rdi, 0xc8(%rsp)
 366d19d:      	movq	0x10(%rbp), %rax
 366d1a1:      	movq	%r9, 0x70(%rsp)
 366d1a6:      	movq	%r9, 0x378(%rsp)
 366d1ae:      	movq	%rax, 0x380(%rsp)
 366d1b6:      	movq	$0x1, 0x280(%rsp)
 366d1c2:      	movq	$0x1, 0x288(%rsp)
 366d1ce:      	movb	$0x0, 0x290(%rsp)
 366d1d6:      	movl	$0x0, 0x200(%rsp)
 366d1e1:      	movb	$0x0, 0x204(%rsp)
 366d1e9:      	movq	$0x0, 0x208(%rsp)
 366d1f5:      	xorps	%xmm0, %xmm0
 366d1f8:      	movaps	%xmm0, 0x100(%rsp)
 366d200:      	movaps	%xmm0, 0x180(%rsp)
 366d208:      	movq	$0x8, 0x210(%rsp)
 366d214:      	movups	%xmm0, 0x218(%rsp)
 366d21c:      	movq	$0x8, 0x228(%rsp)
 366d228:      	movq	$0x0, 0x230(%rsp)
 366d234:      	movb	$0x1, 0x238(%rsp)
 366d23c:      	callq	*0x1ad3b4e(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d242:      	movl	$0x200, %edi            # imm = 0x200
 366d247:      	movl	$0x80, %esi
 366d24c:      	callq	*0x1ad3b46(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d252:      	testq	%rax, %rax
 366d255:      	je	0x366dc98 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xb28>
 366d25b:      	movq	%rax, %r15
 366d25e:      	leaq	0x100(%rsp), %rsi
 366d266:      	movl	$0x200, %edx            # imm = 0x200
 366d26b:      	movq	%rax, %rdi
 366d26e:      	callq	*0x1ad3ac4(%rip)        # 0x5140d38 <writev+0x5140d38>
 366d274:      	movq	$0x1, 0x388(%rsp)
 366d280:      	movq	%r15, 0x390(%rsp)
 366d288:      	movq	$0x1, 0x398(%rsp)
 366d294:      	movq	%r15, 0x38(%rsp)
 366d299:      	movq	%r15, 0x3a0(%rsp)
 366d2a1:      	leaq	0x110(%rsp), %r13
 366d2a9:      	movq	%r13, %rdi
 366d2ac:      	callq	0x388f880 <_RNvXs1_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealthNtNtCs4NRVxsYgnAr_4core7default7Default7default>
 366d2b1:      	movq	$0x1, 0x100(%rsp)
 366d2bd:      	movq	$0x1, 0x108(%rsp)
 366d2c9:      	callq	*0x1ad3ac1(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d2cf:      	movl	$0xe8, %edi
 366d2d4:      	movl	$0x8, %esi
 366d2d9:      	callq	*0x1ad3ab9(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d2df:      	testq	%rax, %rax
 366d2e2:      	je	0x366dbb6 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xa46>
 366d2e8:      	movq	%rax, %r15
 366d2eb:      	leaq	0x100(%rsp), %r13
 366d2f3:      	movl	$0xe8, %edx
 366d2f8:      	movq	%rax, %rdi
 366d2fb:      	movq	%r13, %rsi
 366d2fe:      	callq	*0x1ad3a34(%rip)        # 0x5140d38 <writev+0x5140d38>
 366d304:      	movq	%r15, 0x78(%rsp)
 366d309:      	lock
 366d30a:      	incq	(%r15)
 366d30d:      	jle	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366d313:      	movq	%r15, 0x8(%rsp)
 366d318:      	movq	%r15, 0x130(%rsp)
 366d320:      	movq	$0x1, 0x100(%rsp)
 366d32c:      	movq	0x38(%rsp), %rax
 366d331:      	movq	%rax, 0x108(%rsp)
 366d339:      	movq	0x70(%rsp), %rax
 366d33e:      	movq	%rax, 0x110(%rsp)
 366d346:      	movq	0x10(%rbp), %rax
 366d34a:      	movq	%rax, 0x118(%rsp)
 366d352:      	movq	%r12, 0x138(%rsp)
 366d35a:      	movq	%r14, 0x120(%rsp)
 366d362:      	movl	%ebx, 0x128(%rsp)
 366d369:      	movq	$-0x1, 0xb0(%rsp)
 366d375:      	movq	%r13, (%rsp)
 366d379:      	leaq	0x10(%rsp), %rdi
 366d37e:      	leaq	0xb0(%rsp), %rsi
 366d386:      	xorl	%edx, %edx
 366d388:      	xorl	%r8d, %r8d
 366d38b:      	xorl	%r9d, %r9d
 366d38e:      	callq	0x3c141a0 <_RINvNtNtCs2AWtUsOyxgP_3std6thread9lifecycle15spawn_uncheckedNCNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full0uEB12_>
 366d393:      	movq	0x10(%rsp), %rcx
 366d398:      	movq	0x18(%rsp), %rax
 366d39d:      	testq	%rcx, %rcx
 366d3a0:      	je	0x366dcfb <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xb8b>
 366d3a6:      	movq	0x20(%rsp), %rdx
 366d3ab:      	movq	%rcx, 0x10(%rsp)
 366d3b0:      	movq	%rax, 0x18(%rsp)
 366d3b5:      	movq	%rdx, 0x20(%rsp)
 366d3ba:      	leaq	0x10(%rsp), %rdi
 366d3bf:      	callq	0x3530080 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std6thread11join_handle10JoinHandleuEECs3pwlnhBXFtN_12memra_server>
 366d3c4:      	movq	0x8(%rsp), %rbx
 366d3c9:      	addq	$0x10, %rbx
 366d3cd:      	movl	$0x7d1, %r12d           # imm = 0x7D1
 366d3d3:      	leaq	0x100(%rsp), %r14
 366d3db:      	movq	0x1ad405e(%rip), %r13   # 0x5141440 <writev+0x5141440>
 366d3e2:      	movq	0x1ad3957(%rip), %r15   # 0x5140d40 <writev+0x5140d40>
 366d3e9:      	jmp	0x366d3fa <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x28a>
 366d3eb:      	nopl	(%rax,%rax)
 366d3f0:      	xorl	%edi, %edi
 366d3f2:      	movl	$0xf4240, %esi          # imm = 0xF4240
 366d3f7:      	callq	*%r13
 366d3fa:      	decl	%r12d
 366d3fd:      	je	0x366d42f <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x2bf>
 366d3ff:      	movq	%r14, %rdi
 366d402:      	movq	%rbx, %rsi
 366d405:      	callq	0x3875fe0 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth4live>
 366d40a:      	movq	0x100(%rsp), %rsi
 366d412:      	cmpq	$-0x1, %rsi
 366d416:      	je	0x366d42f <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x2bf>
 366d418:      	testq	%rsi, %rsi
 366d41b:      	je	0x366d3f0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x280>
 366d41d:      	movq	0x108(%rsp), %rdi
 366d425:      	movl	$0x1, %edx
 366d42a:      	callq	*%r15
 366d42d:      	jmp	0x366d3f0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x280>
 366d42f:      	movq	$0x1, 0x3a8(%rsp)
 366d43b:      	movq	0x38(%rsp), %rax
 366d440:      	movq	%rax, 0x3b0(%rsp)
 366d448:      	callq	*0x1ad3942(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d44e:      	movl	$0x18, %edi
 366d453:      	movl	$0x8, %esi
 366d458:      	callq	*0x1ad393a(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d45e:      	testq	%rax, %rax
 366d461:      	je	0x366dbcb <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xa5b>
 366d467:      	movq	%rax, %rbx
 366d46a:      	callq	*0x1ad3920(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d470:      	movl	$0x1, %edi
 366d475:      	movl	$0x1, %esi
 366d47a:      	callq	*0x1ad3918(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d480:      	testq	%rax, %rax
 366d483:      	je	0x366dd27 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbb7>
 366d489:      	movb	$0x6d, (%rax)
 366d48c:      	movq	$0x1, (%rbx)
 366d493:      	movq	%rax, 0x8(%rbx)
 366d497:      	movq	$0x1, 0x10(%rbx)
 366d49f:      	movq	$0x1, 0x100(%rsp)
 366d4ab:      	movq	$0x1, 0x108(%rsp)
 366d4b7:      	movq	$0x1, 0x110(%rsp)
 366d4c3:      	movq	%rbx, 0x118(%rsp)
 366d4cb:      	movq	$0x1, 0x120(%rsp)
 366d4d7:      	callq	*0x1ad38b3(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d4dd:      	movl	$0x28, %edi
 366d4e2:      	movl	$0x8, %esi
 366d4e7:      	callq	*0x1ad38ab(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d4ed:      	testq	%rax, %rax
 366d4f0:      	je	0x366dbe0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xa70>
 366d4f6:      	movq	%rax, %r15
 366d4f9:      	movq	0x120(%rsp), %rax
 366d501:      	movq	%rax, 0x20(%r15)
 366d505:      	movups	0x100(%rsp), %xmm0
 366d50d:      	movups	0x110(%rsp), %xmm1
 366d515:      	movups	%xmm1, 0x10(%r15)
 366d51a:      	movups	%xmm0, (%r15)
 366d51e:      	movq	%r15, 0x80(%rsp)
 366d526:      	movq	0x68(%rsp), %rax
 366d52b:      	movups	(%rax), %xmm0
 366d52e:      	movups	0x10(%rax), %xmm1
 366d532:      	movups	0x20(%rax), %xmm2
 366d536:      	movups	%xmm2, 0x130(%rsp)
 366d53e:      	movups	%xmm1, 0x120(%rsp)
 366d546:      	movups	%xmm0, 0x110(%rsp)
 366d54e:      	movq	$0x1, 0x100(%rsp)
 366d55a:      	movq	$0x1, 0x108(%rsp)
 366d566:      	callq	*0x1ad3824(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d56c:      	movl	$0x40, %edi
 366d571:      	movl	$0x8, %esi
 366d576:      	callq	*0x1ad381c(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d57c:      	testq	%rax, %rax
 366d57f:      	je	0x366dbf5 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xa85>
 366d585:      	movq	%rax, %rbx
 366d588:      	movups	0x100(%rsp), %xmm0
 366d590:      	movups	0x110(%rsp), %xmm1
 366d598:      	movups	0x120(%rsp), %xmm2
 366d5a0:      	movups	0x130(%rsp), %xmm3
 366d5a8:      	movups	%xmm3, 0x30(%rax)
 366d5ac:      	movups	%xmm2, 0x20(%rax)
 366d5b0:      	movups	%xmm1, 0x10(%rax)
 366d5b4:      	movups	%xmm0, (%rax)
 366d5b7:      	movq	%rax, 0x88(%rsp)
 366d5bf:      	movq	$-0x38, %r14
 366d5c6:      	cmpb	$0x1, %fs:0x10(%r14)
 366d5cc:      	jne	0x366dcad <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xb3d>
 366d5d2:      	movq	%fs:(%r14), %rax
 366d5d6:      	movq	%fs:0x8(%r14), %rdx
 366d5db:      	leaq	0x1(%rax), %rcx
 366d5df:      	movq	%rcx, %fs:(%r14)
 366d5e3:      	movq	$0x1, 0x100(%rsp)
 366d5ef:      	movq	$0x1, 0x108(%rsp)
 366d5fb:      	movq	$-0x1, 0x110(%rsp)
 366d607:      	movups	0x1977e2a(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475>
 366d60e:      	movups	%xmm0, 0x188(%rsp)
 366d616:      	movups	0x1977e2b(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475+0x10>
 366d61d:      	movups	%xmm0, 0x198(%rsp)
 366d625:      	movq	%rax, 0x1a8(%rsp)
 366d62d:      	movq	%rdx, 0x1b0(%rsp)
 366d635:      	callq	*0x1ad3755(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d63b:      	movl	$0xb8, %edi
 366d640:      	movl	$0x8, %esi
 366d645:      	callq	*0x1ad374d(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d64b:      	testq	%rax, %rax
 366d64e:      	je	0x366dc12 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xaa2>
 366d654:      	movq	%rax, %r13
 366d657:      	leaq	0x100(%rsp), %rsi
 366d65f:      	movl	$0xb8, %edx
 366d664:      	movq	%rax, %rdi
 366d667:      	callq	*0x1ad36cb(%rip)        # 0x5140d38 <writev+0x5140d38>
 366d66d:      	movq	$0x1, 0x100(%rsp)
 366d679:      	movq	$0x1, 0x108(%rsp)
 366d685:      	movq	$0x0, 0x110(%rsp)
 366d691:      	movb	$0x0, 0x118(%rsp)
 366d699:      	movq	%r13, 0x120(%rsp)
 366d6a1:      	callq	*0x1ad36e9(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d6a7:      	movl	$0x28, %edi
 366d6ac:      	movl	$0x8, %esi
 366d6b1:      	callq	*0x1ad36e1(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d6b7:      	testq	%rax, %rax
 366d6ba:      	je	0x366dc27 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xab7>
 366d6c0:      	movq	%rax, %r12
 366d6c3:      	movq	0x120(%rsp), %rax
 366d6cb:      	movq	%rax, 0x20(%r12)
 366d6d0:      	movups	0x100(%rsp), %xmm0
 366d6d8:      	movups	0x110(%rsp), %xmm1
 366d6e0:      	movups	%xmm1, 0x10(%r12)
 366d6e6:      	movups	%xmm0, (%r12)
 366d6eb:      	movq	%r12, 0x90(%rsp)
 366d6f3:      	movq	$0x0, 0x3b8(%rsp)
 366d6ff:      	movq	$0x0, 0x98(%rsp)
 366d70b:      	movq	$0x0, 0xe0(%rsp)
 366d717:      	movq	$0x0, 0xd0(%rsp)
 366d723:      	movb	$0x0, 0xf8(%rsp)
 366d72b:      	movq	$0x0, 0xe8(%rsp)
 366d737:      	callq	0x3a07a30 <_RNvXsX_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEENtNtCs4NRVxsYgnAr_4core7default7Default7defaultB1y_>
 366d73c:      	movq	%rax, 0x40(%rsp)
 366d741:      	callq	*0x1ad3649(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d747:      	movl	$0x28, %edi
 366d74c:      	movl	$0x8, %esi
 366d751:      	callq	*0x1ad3641(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d757:      	testq	%rax, %rax
 366d75a:      	movq	$-0x38, %r14
 366d761:      	je	0x366dc44 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xad4>
 366d767:      	movq	$0x1, (%rax)
 366d76e:      	movq	$0x1, 0x8(%rax)
 366d776:      	xorps	%xmm0, %xmm0
 366d779:      	movups	%xmm0, 0x10(%rax)
 366d77d:      	movq	$0x0, 0x20(%rax)
 366d785:      	movq	%rax, 0x48(%rsp)
 366d78a:      	cmpb	$0x1, %fs:0x10(%r14)
 366d790:      	jne	0x366dcc7 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xb57>
 366d796:      	movq	%fs:(%r14), %rax
 366d79a:      	movq	%fs:0x8(%r14), %rdx
 366d79f:      	leaq	0x1(%rax), %rcx
 366d7a3:      	movq	%rcx, %fs:(%r14)
 366d7a7:      	movups	0x1977c9a(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475+0x10>
 366d7ae:      	movups	%xmm0, 0x23(%rsp)
 366d7b3:      	movups	0x1977c7e(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475>
 366d7ba:      	movups	%xmm0, 0x13(%rsp)
 366d7bf:      	movq	$0x1, 0x100(%rsp)
 366d7cb:      	movq	$0x1, 0x108(%rsp)
 366d7d7:      	movl	$0x0, 0x110(%rsp)
 366d7e2:      	movb	$0x0, 0x114(%rsp)
 366d7ea:      	movups	0x10(%rsp), %xmm0
 366d7ef:      	movups	%xmm0, 0x115(%rsp)
 366d7f7:      	movups	0x20(%rsp), %xmm0
 366d7fc:      	movups	%xmm0, 0x125(%rsp)
 366d804:      	movl	0x2f(%rsp), %ecx
 366d808:      	movl	%ecx, 0x134(%rsp)
 366d80f:      	movq	%rax, 0x138(%rsp)
 366d817:      	movq	%rdx, 0x140(%rsp)
 366d81f:      	callq	*0x1ad356b(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d825:      	movl	$0x48, %edi
 366d82a:      	movl	$0x8, %esi
 366d82f:      	callq	*0x1ad3563(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d835:      	testq	%rax, %rax
 366d838:      	je	0x366dc59 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xae9>
 366d83e:      	movq	0x140(%rsp), %rcx
 366d846:      	movq	%rcx, 0x40(%rax)
 366d84a:      	movups	0x100(%rsp), %xmm0
 366d852:      	movups	0x110(%rsp), %xmm1
 366d85a:      	movups	0x120(%rsp), %xmm2
 366d862:      	movups	0x130(%rsp), %xmm3
 366d86a:      	movups	%xmm3, 0x30(%rax)
 366d86e:      	movups	%xmm2, 0x20(%rax)
 366d872:      	movups	%xmm1, 0x10(%rax)
 366d876:      	movups	%xmm0, (%rax)
 366d879:      	movq	%rax, 0x50(%rsp)
 366d87e:      	movq	0x8(%rsp), %rax
 366d883:      	movq	%rax, 0x58(%rsp)
 366d888:      	movq	$0x0, 0xb0(%rsp)
 366d894:      	xorl	%edi, %edi
 366d896:      	callq	0x3eae150 <_RNvNtCs3pwlnhBXFtN_12memra_server9audio_api12shared_audio>
 366d89b:      	movq	%rax, 0x60(%rsp)
 366d8a0:      	callq	0x3c6ee00 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store12ttl_from_env>
 366d8a5:      	movq	%rax, %r13
 366d8a8:      	callq	0x3c6f0b0 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store18max_bytes_from_env>
 366d8ad:      	movq	%rax, %r14
 366d8b0:      	xorl	%edi, %edi
 366d8b2:      	callq	0x3c7c600 <_RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 366d8b7:      	testq	%rax, %rax
 366d8ba:      	je	0x366dd39 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbc9>
 366d8c0:      	movups	(%rax), %xmm0
 366d8c3:      	incq	(%rax)
 366d8c6:      	movups	0x19bebbb(%rip), %xmm1  # 0x502c488 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.242.llvm.12140772379724168186+0x10>
 366d8cd:      	movups	%xmm1, 0x23(%rsp)
 366d8d2:      	movups	0x19beb9f(%rip), %xmm1  # 0x502c478 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.242.llvm.12140772379724168186>
 366d8d9:      	movups	%xmm1, 0x13(%rsp)
 366d8de:      	movl	0x2f(%rsp), %eax
 366d8e2:      	movl	%eax, 0x134(%rsp)
 366d8e9:      	movups	0x20(%rsp), %xmm1
 366d8ee:      	movups	%xmm1, 0x125(%rsp)
 366d8f6:      	movups	0x10(%rsp), %xmm1
 366d8fb:      	movups	%xmm1, 0x115(%rsp)
 366d903:      	movq	$0x1, 0x100(%rsp)
 366d90f:      	movq	$0x1, 0x108(%rsp)
 366d91b:      	movl	$0x0, 0x110(%rsp)
 366d926:      	movb	$0x0, 0x114(%rsp)
 366d92e:      	movups	%xmm0, 0x138(%rsp)
 366d936:      	movq	$0x0, 0x148(%rsp)
 366d942:      	movq	%r14, 0x150(%rsp)
 366d94a:      	movq	%r13, 0x158(%rsp)
 366d952:      	movl	$0x0, 0x160(%rsp)
 366d95d:      	callq	*0x1ad342d(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d963:      	movl	$0x68, %edi
 366d968:      	movl	$0x8, %esi
 366d96d:      	callq	*0x1ad3425(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d973:      	testq	%rax, %rax
 366d976:      	je	0x366dc6e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xafe>
 366d97c:      	movq	0x160(%rsp), %rcx
 366d984:      	movq	%rcx, 0x60(%rax)
 366d988:      	movups	0x150(%rsp), %xmm0
 366d990:      	movups	%xmm0, 0x50(%rax)
 366d994:      	movups	0x140(%rsp), %xmm0
 366d99c:      	movups	%xmm0, 0x40(%rax)
 366d9a0:      	movups	0x100(%rsp), %xmm0
 366d9a8:      	movups	0x110(%rsp), %xmm1
 366d9b0:      	movups	0x120(%rsp), %xmm2
 366d9b8:      	movups	0x130(%rsp), %xmm3
 366d9c0:      	movups	%xmm3, 0x30(%rax)
 366d9c4:      	movups	%xmm2, 0x20(%rax)
 366d9c8:      	movups	%xmm1, 0x10(%rax)
 366d9cc:      	movups	%xmm0, (%rax)
 366d9cf:      	movq	%rax, 0xa0(%rsp)
 366d9d7:      	leaq	0x19788f2(%rip), %rax   # 0x4fe62d0 <anon.60c0d423c0cc6e2470573bc6d977e900.615.llvm.14966571587344355475>
 366d9de:      	movq	%rax, 0xa8(%rsp)
 366d9e6:      	movq	$-0x38, %r14
 366d9ed:      	cmpb	$0x1, %fs:0x10(%r14)
 366d9f3:      	jne	0x366dce1 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xb71>
 366d9f9:      	movq	%fs:(%r14), %rax
 366d9fd:      	movq	%fs:0x8(%r14), %rdx
 366da02:      	leaq	0x1(%rax), %rcx
 366da06:      	movq	%rcx, %fs:(%r14)
 366da0a:      	movups	0x1977a37(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475+0x10>
 366da11:      	movups	%xmm0, 0x23(%rsp)
 366da16:      	movups	0x1977a1b(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475>
 366da1d:      	movups	%xmm0, 0x13(%rsp)
 366da22:      	movq	$0x1, 0x100(%rsp)
 366da2e:      	movq	$0x1, 0x108(%rsp)
 366da3a:      	movl	$0x0, 0x110(%rsp)
 366da45:      	movb	$0x0, 0x114(%rsp)
 366da4d:      	movups	0x10(%rsp), %xmm0
 366da52:      	movups	%xmm0, 0x115(%rsp)
 366da5a:      	movups	0x20(%rsp), %xmm0
 366da5f:      	movups	%xmm0, 0x125(%rsp)
 366da67:      	movl	0x2f(%rsp), %ecx
 366da6b:      	movl	%ecx, 0x134(%rsp)
 366da72:      	movq	%rax, 0x138(%rsp)
 366da7a:      	movq	%rdx, 0x140(%rsp)
 366da82:      	callq	*0x1ad3308(%rip)        # 0x5140d90 <writev+0x5140d90>
 366da88:      	movl	$0x48, %edi
 366da8d:      	movl	$0x8, %esi
 366da92:      	callq	*0x1ad3300(%rip)        # 0x5140d98 <writev+0x5140d98>
 366da98:      	testq	%rax, %rax
 366da9b:      	je	0x366dc83 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xb13>
 366daa1:      	movq	0x140(%rsp), %rcx
 366daa9:      	movq	%rcx, 0x40(%rax)
 366daad:      	movups	0x100(%rsp), %xmm0
 366dab5:      	movups	0x110(%rsp), %xmm1
 366dabd:      	movups	0x120(%rsp), %xmm2
 366dac5:      	movups	0x130(%rsp), %xmm3
 366dacd:      	movups	%xmm3, 0x30(%rax)
 366dad1:      	movups	%xmm2, 0x20(%rax)
 366dad5:      	movups	%xmm1, 0x10(%rax)
 366dad9:      	movups	%xmm0, (%rax)
 366dadc:      	movq	0xc8(%rsp), %rdx
 366dae4:      	movq	$0x1, (%rdx)
 366daeb:      	movq	0x38(%rsp), %rcx
 366daf0:      	movq	%rcx, 0x8(%rdx)
 366daf4:      	movq	%r15, 0x28(%rdx)
 366daf8:      	movq	%rbx, 0x30(%rdx)
 366dafc:      	movq	%r12, 0x38(%rdx)
 366db00:      	movq	$0x0, 0x80(%rdx)
 366db0b:      	movq	$0x0, 0x90(%rdx)
 366db16:      	movq	0xe0(%rsp), %rcx
 366db1e:      	movq	%rcx, 0xa8(%rdx)
 366db25:      	movups	0xd0(%rsp), %xmm0
 366db2d:      	movups	%xmm0, 0x98(%rdx)
 366db34:      	movups	0xe8(%rsp), %xmm0
 366db3c:      	movups	%xmm0, 0x10(%rdx)
 366db40:      	movq	0xf8(%rsp), %rcx
 366db48:      	movq	%rcx, 0x20(%rdx)
 366db4c:      	movq	0x40(%rsp), %rcx
 366db51:      	movq	%rcx, 0x40(%rdx)
 366db55:      	movq	0x48(%rsp), %rcx
 366db5a:      	movq	%rcx, 0x48(%rdx)
 366db5e:      	movq	0x50(%rsp), %rcx
 366db63:      	movq	%rcx, 0x50(%rdx)
 366db67:      	movq	0x58(%rsp), %rcx
 366db6c:      	movq	%rcx, 0x58(%rdx)
 366db70:      	movups	0xb0(%rsp), %xmm0
 366db78:      	movups	%xmm0, 0xb0(%rdx)
 366db7f:      	movq	0xc0(%rsp), %rcx
 366db87:      	movq	%rcx, 0xc0(%rdx)
 366db8e:      	movq	0x60(%rsp), %rcx
 366db93:      	movq	%rcx, 0x60(%rdx)
 366db97:      	movaps	0xa0(%rsp), %xmm0
 366db9f:      	movups	%xmm0, 0x68(%rdx)
 366dba3:      	movq	%rax, 0x78(%rdx)
 366dba7:      	leaq	-0x28(%rbp), %rsp
 366dbab:      	popq	%rbx
 366dbac:      	popq	%r12
 366dbae:      	popq	%r13
 366dbb0:      	popq	%r14
 366dbb2:      	popq	%r15
 366dbb4:      	popq	%rbp
 366dbb5:      	retq
 366dbb6:      	movl	$0x8, %edi
 366dbbb:      	movl	$0xe8, %esi
 366dbc0:      	callq	*0x1ad339a(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dbc6:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dbcb:      	movl	$0x8, %edi
 366dbd0:      	movl	$0x18, %esi
 366dbd5:      	callq	*0x1ad3385(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dbdb:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dbe0:      	movl	$0x8, %edi
 366dbe5:      	movl	$0x28, %esi
 366dbea:      	callq	*0x1ad3370(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dbf0:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dbf5:      	leaq	0x110(%rsp), %rbx
 366dbfd:      	movl	$0x8, %edi
 366dc02:      	movl	$0x40, %esi
 366dc07:      	callq	*0x1ad3353(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc0d:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc12:      	movl	$0x8, %edi
 366dc17:      	movl	$0xb8, %esi
 366dc1c:      	callq	*0x1ad333e(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc22:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc27:      	leaq	0x120(%rsp), %rbx
 366dc2f:      	movl	$0x8, %edi
 366dc34:      	movl	$0x28, %esi
 366dc39:      	callq	*0x1ad3321(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc3f:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc44:      	movl	$0x8, %edi
 366dc49:      	movl	$0x28, %esi
 366dc4e:      	callq	*0x1ad330c(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc54:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc59:      	movl	$0x8, %edi
 366dc5e:      	movl	$0x48, %esi
 366dc63:      	callq	*0x1ad32f7(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc69:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc6e:      	movl	$0x8, %edi
 366dc73:      	movl	$0x68, %esi
 366dc78:      	callq	*0x1ad32e2(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc7e:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc83:      	movl	$0x8, %edi
 366dc88:      	movl	$0x48, %esi
 366dc8d:      	callq	*0x1ad32cd(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc93:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dc98:      	movl	$0x80, %edi
 366dc9d:      	movl	$0x200, %esi            # imm = 0x200
 366dca2:      	callq	*0x1ad32b8(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dca8:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dcad:      	callq	*0x1ad34e5(%rip)        # 0x5141198 <writev+0x5141198>
 366dcb3:      	movq	%rax, %fs:(%r14)
 366dcb7:      	movq	%rdx, %fs:0x8(%r14)
 366dcbc:      	movb	$0x1, %fs:0x10(%r14)
 366dcc2:      	jmp	0x366d5db <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x46b>
 366dcc7:      	callq	*0x1ad34cb(%rip)        # 0x5141198 <writev+0x5141198>
 366dccd:      	movq	%rax, %fs:(%r14)
 366dcd1:      	movq	%rdx, %fs:0x8(%r14)
 366dcd6:      	movb	$0x1, %fs:0x10(%r14)
 366dcdc:      	jmp	0x366d79f <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x62f>
 366dce1:      	callq	*0x1ad34b1(%rip)        # 0x5141198 <writev+0x5141198>
 366dce7:      	movq	%rax, %fs:(%r14)
 366dceb:      	movq	%rdx, %fs:0x8(%r14)
 366dcf0:      	movb	$0x1, %fs:0x10(%r14)
 366dcf6:      	jmp	0x366da02 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0x892>
 366dcfb:      	movq	%rax, 0x10(%rsp)
 366dd00:      	leaq	-0x3266de2(%rip), %rdi  # 0x406f25 <anon.60c0d423c0cc6e2470573bc6d977e900.3554.llvm.14966571587344355475+0x242>
 366dd07:      	leaq	0x198183a(%rip), %rcx   # 0x4fef548 <anon.60c0d423c0cc6e2470573bc6d977e900.3560.llvm.14966571587344355475>
 366dd0e:      	leaq	0x1981f83(%rip), %r8    # 0x4fefc98 <anon.60c0d423c0cc6e2470573bc6d977e900.3581.llvm.14966571587344355475+0x4b0>
 366dd15:      	leaq	0x10(%rsp), %rdx
 366dd1a:      	movl	$0x16, %esi
 366dd1f:      	callq	*0x1ad3173(%rip)        # 0x5140e98 <writev+0x5140e98>
 366dd25:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dd27:      	movl	$0x1, %edi
 366dd2c:      	movl	$0x1, %esi
 366dd31:      	callq	*0x1ad3051(%rip)        # 0x5140d88 <writev+0x5140d88>
 366dd37:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xbd6>
 366dd39:      	leaq	0x19a9d90(%rip), %rdi   # 0x5017ad0 <anon.090efebfaa215158d04f118ce3d95888.2.llvm.13030924631975612447>
 366dd40:      	callq	*0x1ad442a(%rip)        # 0x5142170 <writev+0x5142170>
 366dd46:      	ud2
 366dd48:      	movq	%rax, %r15
 366dd4b:      	jmp	0x366ddf1 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xc81>
 366dd50:      	movq	%rax, %r15
 366dd53:      	jmp	0x366de9e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd2e>
 366dd58:      	movq	%rax, %r15
 366dd5b:      	jmp	0x366dfa5 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe35>
 366dd60:      	movq	%rax, %r15
 366dd63:      	jmp	0x366de39 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xcc9>
 366dd68:      	movq	%rax, %r15
 366dd6b:      	movb	$0x1, %r14b
 366dd6e:      	jmp	0x366ded9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd69>
 366dd73:      	movq	%rax, %r15
 366dd76:      	movl	$0x18, %esi
 366dd7b:      	movl	$0x8, %edx
 366dd80:      	movq	%rbx, %rdi
 366dd83:      	callq	*0x1ad2fb7(%rip)        # 0x5140d40 <writev+0x5140d40>
 366dd89:      	jmp	0x366e013 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xea3>
 366dd8e:      	movq	%rax, %r15
 366dd91:      	leaq	0x10(%rsp), %rdi
 366dd96:      	callq	0x3550de0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 366dd9b:      	jmp	0x366ddb5 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xc45>
 366dd9d:      	callq	*0x1ad3015(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dda3:      	movq	%rax, %r15
 366dda6:      	jmp	0x366e057 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xee7>
 366ddab:      	jmp	0x366ddb2 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xc42>
 366ddad:      	movq	%rax, %r15
 366ddb0:      	jmp	0x366de24 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xcb4>
 366ddb2:      	movq	%rax, %r15
 366ddb5:      	movb	$0x1, %r14b
 366ddb8:      	movb	$0x1, %bl
 366ddba:      	movq	0x8(%rsp), %rax
 366ddbf:      	jmp	0x366e032 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xec2>
 366ddc4:      	movq	%rax, %r15
 366ddc7:      	leaq	0x100(%rsp), %rdi
 366ddcf:      	callq	0x3c04350 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync4mpmc7counter7CounterINtNtBG_4list7ChannelNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdEEEB1R_.llvm.12140772379724168186>
 366ddd4:      	movb	$0x1, %bl
 366ddd6:      	jmp	0x366e07b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xf0b>
 366dddb:      	callq	*0x1ad2fd7(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dde1:      	movq	%rax, %r15
 366dde4:      	leaq	0x118(%rsp), %rdi
 366ddec:      	callq	0x392d530 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server15background_jobs7ControlEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366ddf1:      	movq	0xa0(%rsp), %rax
 366ddf9:      	lock
 366ddfa:      	decq	(%rax)
 366ddfd:      	jne	0x366de24 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xcb4>
 366ddff:      	leaq	0xa0(%rsp), %rdi
 366de07:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 366de0c:      	jmp	0x366de24 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xcb4>
 366de0e:      	callq	*0x1ad2fa4(%rip)        # 0x5140db8 <writev+0x5140db8>
 366de14:      	movq	%rax, %r15
 366de17:      	leaq	0x118(%rsp), %rdi
 366de1f:      	callq	0x392dfe0 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366de24:      	movq	0x60(%rsp), %rax
 366de29:      	lock
 366de2a:      	decq	(%rax)
 366de2d:      	jne	0x366de39 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xcc9>
 366de2f:      	leaq	0x60(%rsp), %rdi
 366de34:      	callq	0x39bbe80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCsdMwdNqnNPrU_11memra_lanes12audio_stream14AudioSchedulerEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366de39:      	movq	0xb0(%rsp), %rax
 366de41:      	testq	%rax, %rax
 366de44:      	je	0x366de59 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xce9>
 366de46:      	lock
 366de47:      	decq	(%rax)
 366de4a:      	jne	0x366de59 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xce9>
 366de4c:      	leaq	0xb0(%rsp), %rdi
 366de54:      	callq	0x39bd080 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server8darklane10BgJobStateE9drop_slowBJ_>
 366de59:      	movq	0x58(%rsp), %rax
 366de5e:      	lock
 366de5f:      	decq	(%rax)
 366de62:      	jne	0x366de6e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xcfe>
 366de64:      	leaq	0x58(%rsp), %rdi
 366de69:      	callq	0x39bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 366de6e:      	movq	0x50(%rsp), %rax
 366de73:      	lock
 366de74:      	decq	(%rax)
 366de77:      	jne	0x366de83 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd13>
 366de79:      	leaq	0x50(%rsp), %rdi
 366de7e:      	callq	0x39bb4f0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringjEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366de83:      	xorl	%r14d, %r14d
 366de86:      	jmp	0x366dea1 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd31>
 366de88:      	callq	*0x1ad2f2a(%rip)        # 0x5140db8 <writev+0x5140db8>
 366de8e:      	movq	%rax, %r15
 366de91:      	leaq	0x118(%rsp), %rdi
 366de99:      	callq	0x392e950 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringyEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropCs3pwlnhBXFtN_12memra_server>
 366de9e:      	movb	$0x1, %r14b
 366dea1:      	movq	0x48(%rsp), %rax
 366dea6:      	lock
 366dea7:      	decq	(%rax)
 366deaa:      	jne	0x366dec4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd54>
 366deac:      	leaq	0x48(%rsp), %rdi
 366deb1:      	callq	0x39b9f80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6worker15EventQueueStateE9drop_slowBJ_>
 366deb6:      	jmp	0x366dec4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd54>
 366deb8:      	callq	*0x1ad2efa(%rip)        # 0x5140db8 <writev+0x5140db8>
 366debe:      	movq	%rax, %r15
 366dec1:      	movb	$0x1, %r14b
 366dec4:      	movq	0x40(%rsp), %rax
 366dec9:      	lock
 366deca:      	decq	(%rax)
 366decd:      	jne	0x366ded9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd69>
 366decf:      	leaq	0x40(%rsp), %rdi
 366ded4:      	callq	0x39bb8c0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEE9drop_slowB1y_>
 366ded9:      	movq	0xe8(%rsp), %rax
 366dee1:      	testq	%rax, %rax
 366dee4:      	je	0x366defa <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd8a>
 366dee6:      	lock
 366dee7:      	decq	(%rax)
 366deea:      	jne	0x366defa <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xd8a>
 366deec:      	leaq	0xe8(%rsp), %rdi
 366def4:      	callq	*0x1ad40ae(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 366defa:      	movq	0xd0(%rsp), %rax
 366df02:      	testq	%rax, %rax
 366df05:      	je	0x366df1b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xdab>
 366df07:      	lock
 366df08:      	decq	(%rax)
 366df0b:      	jne	0x366df1b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xdab>
 366df0d:      	leaq	0xd0(%rsp), %rdi
 366df15:      	callq	*0x1ad408d(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 366df1b:      	movq	0x98(%rsp), %rax
 366df23:      	testq	%rax, %rax
 366df26:      	je	0x366df3b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xdcb>
 366df28:      	lock
 366df29:      	decq	(%rax)
 366df2c:      	jne	0x366df3b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xdcb>
 366df2e:      	leaq	0x98(%rsp), %rdi
 366df36:      	callq	0x39bb070 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringIBw_NtCscIVK9LJ3de3_15memra_tokenizer9TokenizerEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366df3b:      	movq	0x3b8(%rsp), %rax
 366df43:      	testq	%rax, %rax
 366df46:      	je	0x366df5b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xdeb>
 366df48:      	lock
 366df49:      	decq	(%rax)
 366df4c:      	jne	0x366df5b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xdeb>
 366df4e:      	leaq	0x3b8(%rsp), %rdi
 366df56:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 366df5b:      	movq	0x90(%rsp), %rax
 366df63:      	lock
 366df64:      	decq	(%rax)
 366df67:      	jne	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe38>
 366df69:      	leaq	0x90(%rsp), %rdi
 366df71:      	callq	0x39bc090 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockIBw_NtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEE9drop_slowB1C_>
 366df76:      	jmp	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe38>
 366df78:      	movq	%rax, %r15
 366df7b:      	lock
 366df7c:      	decq	(%r13)
 366df80:      	movb	$0x1, %r14b
 366df83:      	jne	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe38>
 366df85:      	movq	%rbx, %rdi
 366df88:      	callq	0x39bc730 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetE9drop_slowBH_>
 366df8d:      	jmp	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe38>
 366df8f:      	callq	*0x1ad2e23(%rip)        # 0x5140db8 <writev+0x5140db8>
 366df95:      	movq	%rax, %r15
 366df98:      	leaq	0x100(%rsp), %rdi
 366dfa0:      	callq	0x352f670 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCscdodAO9FK5_5alloc4sync8ArcInnerNtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEB1f_>
 366dfa5:      	movb	$0x1, %r14b
 366dfa8:      	movq	0x88(%rsp), %rax
 366dfb0:      	lock
 366dfb1:      	decq	(%rax)
 366dfb4:      	jne	0x366dfd9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe69>
 366dfb6:      	leaq	0x88(%rsp), %rdi
 366dfbe:      	callq	0x39bb0e0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEE9drop_slowB1Z_>
 366dfc3:      	jmp	0x366dfd9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe69>
 366dfc5:      	callq	*0x1ad2ded(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dfcb:      	movq	%rax, %r15
 366dfce:      	movq	%rbx, %rdi
 366dfd1:      	callq	0x392de70 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366dfd6:      	movb	$0x1, %r14b
 366dfd9:      	movq	0x80(%rsp), %rax
 366dfe1:      	lock
 366dfe2:      	decq	(%rax)
 366dfe5:      	jne	0x366dff4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xe84>
 366dfe7:      	leaq	0x80(%rsp), %rdi
 366dfef:      	callq	0x39ba040 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtB7_3vec3VecNtNtB7_6string6StringEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366dff4:      	xorl	%ebx, %ebx
 366dff6:      	jmp	0x366e018 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xea8>
 366dff8:      	callq	*0x1ad2dba(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dffe:      	movq	%rax, %r15
 366e001:      	leaq	0x100(%rsp), %rdi
 366e009:      	callq	0x352f3d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCscdodAO9FK5_5alloc4sync8ArcInnerINtNtBG_3vec3VecNtNtBG_6string6StringEEECs3pwlnhBXFtN_12memra_server>
 366e00e:      	jmp	0x366e013 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xea3>
 366e010:      	movq	%rax, %r15
 366e013:      	movb	$0x1, %bl
 366e015:      	movb	$0x1, %r14b
 366e018:      	leaq	0x3a8(%rsp), %rdi
 366e020:      	callq	0x3c723a0 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 366e025:      	testb	%r14b, %r14b
 366e028:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xf20>
 366e02a:      	movq	0x78(%rsp), %rax
 366e02f:      	xorl	%r14d, %r14d
 366e032:      	lock
 366e033:      	decq	(%rax)
 366e036:      	jne	0x366e042 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xed2>
 366e038:      	leaq	0x78(%rsp), %rdi
 366e03d:      	callq	0x39bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 366e042:      	testb	%r14b, %r14b
 366e045:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xf20>
 366e047:      	xorl	%r14d, %r14d
 366e04a:      	jmp	0x366e069 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xef9>
 366e04c:      	movq	%rax, %r15
 366e04f:      	movq	%r13, %rdi
 366e052:      	callq	0x384d3c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthEBF_.llvm.5209651971967471703>
 366e057:      	movb	$0x1, %bl
 366e059:      	leaq	0x398(%rsp), %rdi
 366e061:      	callq	0x3c79310 <_RNvXsi_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_8ReceiverNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBU_>
 366e066:      	movb	$0x1, %r14b
 366e069:      	leaq	0x388(%rsp), %rdi
 366e071:      	callq	0x3c723a0 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 366e076:      	testb	%r14b, %r14b
 366e079:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xf20>
 366e07b:      	cmpq	$-0x1, 0x70(%rsp)
 366e081:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xf20>
 366e083:      	leaq	0x378(%rsp), %rdi
 366e08b:      	callq	0x3c71d20 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server5tests9WorkerSawENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 366e090:      	testb	%bl, %bl
 366e092:      	je	0x366e09e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475+0xf2e>
 366e094:      	movq	0x68(%rsp), %rdi
 366e099:      	callq	0x392de70 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366e09e:      	movq	%r15, %rdi
 366e0a1:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 366e0a6:      	callq	*0x1ad2d0c(%rip)        # 0x5140db8 <writev+0x5140db8>
 366e0ac:      	callq	*0x1ad2d06(%rip)        # 0x5140db8 <writev+0x5140db8>
 366e0b2:      	int3
 366e0b3:      	int3
 366e0b4:      	int3
 366e0b5:      	int3
 366e0b6:      	int3
 366e0b7:      	int3
 366e0b8:      	int3
 366e0b9:      	int3
 366e0ba:      	int3
 366e0bb:      	int3
 366e0bc:      	int3
 366e0bd:      	int3
 366e0be:      	int3
 366e0bf:      	int3

000000000366f700 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention>:
 366f700:      	pushq	%rbp
 366f701:      	pushq	%r15
 366f703:      	pushq	%r14
 366f705:      	pushq	%r13
 366f707:      	pushq	%r12
 366f709:      	pushq	%rbx
 366f70a:      	subq	$0x158, %rsp            # imm = 0x158
 366f711:      	movq	%rcx, %rbx
 366f714:      	movq	%rdx, %r15
 366f717:      	movq	%rsi, %r12
 366f71a:      	movq	%rdi, 0x10(%rsp)
 366f71f:      	callq	*0x1ad29bb(%rip)        # 0x51420e0 <writev+0x51420e0>
 366f725:      	imulq	$0x10624dd3, %rbx, %r13 # imm = 0x10624DD3
 366f72c:      	shrq	$0x26, %r13
 366f730:      	imull	$0x3e8, %r13d, %ecx     # imm = 0x3E8
 366f737:      	movl	%ebx, %esi
 366f739:      	subl	%ecx, %esi
 366f73b:      	imull	$0xf4240, %esi, %ecx    # imm = 0xF4240
 366f741:      	movq	%rax, %rdi
 366f744:      	movl	%edx, %esi
 366f746:      	movq	%r13, %rdx
 366f749:      	movl	%ecx, 0xc(%rsp)
 366f74d:      	callq	*0x1ad2995(%rip)        # 0x51420e8 <writev+0x51420e8>
 366f753:      	movq	%rax, 0xd0(%rsp)
 366f75b:      	movl	%edx, 0xd8(%rsp)
 366f762:      	movq	%rbx, 0x18(%rsp)
 366f767:      	movq	%rbx, 0xc8(%rsp)
 366f76f:      	movl	0x1ade9bb(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451+0x8>
 366f775:      	testl	%eax, %eax
 366f777:      	jne	0x366f950 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x250>
 366f77d:      	movq	0x1ade9a4(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 366f784:      	leaq	0x1ae3f2d(%rip), %rax   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 366f78b:      	leaq	0x1ae3f0e(%rip), %r10   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 366f792:      	leaq	0x38(%rsp), %rdi
 366f797:      	leaq	0xc8(%rsp), %rbx
 366f79f:      	movq	%r12, %rbp
 366f7a2:      	movq	%r12, %rsi
 366f7a5:      	xorl	%edx, %edx
 366f7a7:      	movq	%r15, %r12
 366f7aa:      	movq	%r15, %rcx
 366f7ad:      	movq	%rbx, %r8
 366f7b0:      	pushq	%rax
 366f7b1:      	pushq	%r10
 366f7b3:      	callq	0x3ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 366f7b8:      	addq	$0x10, %rsp
 366f7bc:      	movl	$0x32, %r15d
 366f7c2:      	jmp	0x366f7f9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0xf9>
 366f7c4:      	nopw	%cs:(%rax,%rax)
 366f7d0:      	leaq	0x40(%rsp), %rdi
 366f7d5:      	callq	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 366f7da:      	leaq	0x38(%rsp), %r14
 366f7df:      	movl	$0x90, %edx
 366f7e4:      	movq	%r14, %rdi
 366f7e7:      	movq	%rbx, %rsi
 366f7ea:      	callq	*0x1ad1548(%rip)        # 0x5140d38 <writev+0x5140d38>
 366f7f0:      	decl	%r15d
 366f7f3:      	je	0x366f8e2 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x1e2>
 366f7f9:      	cmpl	$-0x1, 0x38(%rsp)
 366f7fe:      	je	0x366f8e2 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x1e2>
 366f804:      	cmpq	$0xd, 0xc0(%rsp)
 366f80d:      	jne	0x366f909 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x209>
 366f813:      	movq	0xb8(%rsp), %rax
 366f81b:      	movq	(%rax), %rcx
 366f81e:      	movabsq	$0x6165645f64656873, %rdx # imm = 0x6165645F64656873
 366f828:      	xorq	%rdx, %rcx
 366f82b:      	movq	0x5(%rax), %rax
 366f82f:      	movabsq	$0x656e696c64616564, %rdx # imm = 0x656E696C64616564
 366f839:      	xorq	%rdx, %rax
 366f83c:      	orq	%rcx, %rax
 366f83f:      	jne	0x366f909 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x209>
 366f845:      	xorl	%edi, %edi
 366f847:      	movl	$0x989680, %esi         # imm = 0x989680
 366f84c:      	callq	*0x1ad1bee(%rip)        # 0x5141440 <writev+0x5141440>
 366f852:      	callq	*0x1ad2888(%rip)        # 0x51420e0 <writev+0x51420e0>
 366f858:      	movq	%rax, %rdi
 366f85b:      	movl	%edx, %esi
 366f85d:      	movq	%r13, %rdx
 366f860:      	movl	0xc(%rsp), %ecx
 366f864:      	callq	*0x1ad287e(%rip)        # 0x51420e8 <writev+0x51420e8>
 366f86a:      	movq	%rax, 0x28(%rsp)
 366f86f:      	movl	%edx, 0x30(%rsp)
 366f873:      	movq	0x18(%rsp), %rax
 366f878:      	movq	%rax, 0x20(%rsp)
 366f87d:      	movl	0x1ade8ad(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451+0x8>
 366f883:      	testl	%eax, %eax
 366f885:      	jne	0x366f8d4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x1d4>
 366f887:      	movq	0x1ade89a(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 366f88e:      	movq	%rbx, %rdi
 366f891:      	movq	%rbp, %rsi
 366f894:      	xorl	%edx, %edx
 366f896:      	movq	%r12, %rcx
 366f899:      	leaq	0x20(%rsp), %r8
 366f89e:      	leaq	0x1ae3e13(%rip), %rax   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 366f8a5:      	pushq	%rax
 366f8a6:      	leaq	0x1ae3df3(%rip), %rax   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 366f8ad:      	pushq	%rax
 366f8ae:      	callq	0x3ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 366f8b3:      	addq	$0x10, %rsp
 366f8b7:      	cmpl	$-0x1, 0x38(%rsp)
 366f8bc:      	je	0x366f7d0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0xd0>
 366f8c2:      	leaq	0x38(%rsp), %r14
 366f8c7:      	movq	%r14, %rdi
 366f8ca:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 366f8cf:      	jmp	0x366f7df <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0xdf>
 366f8d4:      	leaq	0x1ade84d(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 366f8db:      	callq	0x3b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 366f8e0:      	jmp	0x366f887 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x187>
 366f8e2:      	leaq	0x38(%rsp), %rsi
 366f8e7:      	movl	$0x90, %edx
 366f8ec:      	movq	0x10(%rsp), %rdi
 366f8f1:      	callq	*0x1ad1441(%rip)        # 0x5140d38 <writev+0x5140d38>
 366f8f7:      	addq	$0x158, %rsp            # imm = 0x158
 366f8fe:      	popq	%rbx
 366f8ff:      	popq	%r12
 366f901:      	popq	%r13
 366f903:      	popq	%r14
 366f905:      	popq	%r15
 366f907:      	popq	%rbp
 366f908:      	retq
 366f909:      	leaq	0xb8(%rsp), %rax
 366f911:      	movq	%rax, 0x20(%rsp)
 366f916:      	leaq	0x20(%rsp), %rax
 366f91b:      	movq	%rax, 0xc8(%rsp)
 366f923:      	leaq	0x396326(%rip), %rax    # 0x3a05c50 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtRReNtB6_7Display3fmtCs3pwlnhBXFtN_12memra_server>
 366f92a:      	movq	%rax, 0xd0(%rsp)
 366f932:      	leaq	-0x32dd295(%rip), %rdi  # 0x3926a4 <anon.c030ccd08983dd522ef199ffd40d4bd6.208.llvm.15771381525616154697+0x24>
 366f939:      	leaq	0x1980698(%rip), %rdx   # 0x4feffd8 <anon.60c0d423c0cc6e2470573bc6d977e900.3581.llvm.14966571587344355475+0x7f0>
 366f940:      	leaq	0xc8(%rsp), %rsi
 366f948:      	callq	*0x1ad1532(%rip)        # 0x5140e80 <writev+0x5140e80>
 366f94e:      	ud2
 366f950:      	leaq	0x1ade7d1(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 366f957:      	callq	0x3b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 366f95c:      	jmp	0x366f77d <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x7d>
 366f961:      	movq	%rax, %rbx
 366f964:      	leaq	0x38(%rsp), %rdi
 366f969:      	leaq	0xc8(%rsp), %rsi
 366f971:      	movl	$0x90, %edx
 366f976:      	callq	*0x1ad13bc(%rip)        # 0x5140d38 <writev+0x5140d38>
 366f97c:      	jmp	0x366f983 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x283>
 366f97e:      	jmp	0x366f980 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x280>
 366f980:      	movq	%rax, %rbx
 366f983:      	leaq	0x38(%rsp), %rdi
 366f988:      	callq	0x352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>
 366f98d:      	movq	%rbx, %rdi
 366f990:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 366f995:      	callq	*0x1ad141d(%rip)        # 0x5140db8 <writev+0x5140db8>
 366f99b:      	int3
 366f99c:      	int3
 366f99d:      	int3
 366f99e:      	int3
 366f99f:      	int3

000000000367dfb0 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped>:
 367dfb0:      	pushq	%r15
 367dfb2:      	pushq	%r14
 367dfb4:      	pushq	%rbx
 367dfb5:      	subq	$0x260, %rsp            # imm = 0x260
 367dfbc:      	movl	$0x1, %ecx
 367dfc1:      	xorl	%eax, %eax
 367dfc3:      	lock
 367dfc4:      	cmpxchgl	%ecx, 0x1ad56f5(%rip)   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 367dfcb:      	jne	0x367e3be <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x40e>
 367dfd1:      	movq	0x1ac3090(%rip), %r14   # 0x5141068 <writev+0x5141068>
 367dfd8:      	movq	(%r14), %rax
 367dfdb:      	shlq	%rax
 367dfde:      	testq	%rax, %rax
 367dfe1:      	jne	0x367e3d0 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x420>
 367dfe7:      	xorl	%ebx, %ebx
 367dfe9:      	movzbl	0x1ad56d4(%rip), %eax   # 0x51536c4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK+0x4>
 367dff0:      	testb	%al, %al
 367dff2:      	je	0x367dffb <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4b>
 367dff4:      	movb	$0x0, 0x1ad56c9(%rip)   # 0x51536c4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK+0x4>
 367dffb:      	xorl	%eax, %eax
 367dffd:      	xchgb	%al, 0x1ad5706(%rip)    # 0x5153709 <_RNvCs3pwlnhBXFtN_12memra_server8DRAINING>
 367e003:      	movl	$0x1, %ecx
 367e008:      	xorl	%eax, %eax
 367e00a:      	lock
 367e00b:      	cmpxchgl	%ecx, 0x1ad56ce(%rip)   # 0x51536e0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS>
 367e012:      	jne	0x367e3ef <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x43f>
 367e018:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 367e022:      	movq	(%r14), %rcx
 367e025:      	testq	%rax, %rcx
 367e028:      	jne	0x367e401 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x451>
 367e02e:      	xorl	%eax, %eax
 367e030:      	movzbl	0x1ad56ad(%rip), %ecx   # 0x51536e4 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS+0x4>
 367e037:      	leaq	0x1ad56a2(%rip), %rcx   # 0x51536e0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS>
 367e03e:      	movq	%rcx, 0x20(%rsp)
 367e043:      	movb	%al, 0x28(%rsp)
 367e047:      	leaq	0x1ad5672(%rip), %rax   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 367e04e:      	movq	%rax, 0x30(%rsp)
 367e053:      	movb	%bl, 0x38(%rsp)
 367e057:      	movq	$-0x38, %rbx
 367e05e:      	cmpb	$0x1, %fs:0x10(%rbx)
 367e063:      	jne	0x367e40e <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x45e>
 367e069:      	movq	%fs:(%rbx), %rax
 367e06d:      	movq	%fs:0x8(%rbx), %rdx
 367e072:      	leaq	0x1(%rax), %rcx
 367e076:      	movq	%rcx, %fs:(%rbx)
 367e07a:      	movups	0x19673b7(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475>
 367e081:      	movaps	%xmm0, 0x40(%rsp)
 367e086:      	movups	0x19673bb(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475+0x10>
 367e08d:      	movaps	%xmm0, 0x50(%rsp)
 367e092:      	movq	%rax, 0x60(%rsp)
 367e097:      	movq	%rdx, 0x68(%rsp)
 367e09c:      	subq	$0x10, %rsp
 367e0a0:      	leaq	0x1a8(%rsp), %rdi
 367e0a8:      	leaq	0x50(%rsp), %r8
 367e0ad:      	movl	$0x1, %esi
 367e0b2:      	xorl	%edx, %edx
 367e0b4:      	xorl	%ecx, %ecx
 367e0b6:      	movq	$-0x1, %r9
 367e0bd:      	callq	0x366d170 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475>
 367e0c2:      	addq	$0x10, %rsp
 367e0c6:      	movl	0x1ad00bc(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x18>
 367e0cc:      	testl	%eax, %eax
 367e0ce:      	jne	0x367e427 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x477>
 367e0d4:      	movq	0x1ad00a5(%rip), %rbx   # 0x514e180 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x10>
 367e0db:      	movl	0x1acffff(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x10>
 367e0e1:      	testl	%eax, %eax
 367e0e3:      	jne	0x367e438 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x488>
 367e0e9:      	movq	0x1acffe0(%rip), %rcx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 367e0f0:      	movq	0x1acffe1(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x8>
 367e0f7:      	movq	%rbx, %rdx
 367e0fa:      	shrq	$0x3e, %rdx
 367e0fe:      	jne	0x367e449 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x499>
 367e104:      	shlq	$0x2, %rbx
 367e108:      	testq	%rcx, %rcx
 367e10b:      	cmovneq	%rax, %rbx
 367e10f:      	movq	%rbx, 0x1ad559a(%rip)   # 0x51536b0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS+0x10>
 367e116:      	movq	$0x0, 0x1ad557f(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 367e121:      	movl	0x1ad0061(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x18>
 367e127:      	testl	%eax, %eax
 367e129:      	jne	0x367e455 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4a5>
 367e12f:      	movq	0x1ad003a(%rip), %rax   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 367e136:      	movq	%rax, (%rsp)
 367e13a:      	movq	$0x1, 0x8(%rsp)
 367e143:      	leaq	0x18(%rsp), %rbx
 367e148:      	xorps	%xmm0, %xmm0
 367e14b:      	movups	%xmm0, 0x10(%rsp)
 367e150:      	callq	*0x1ac3f8a(%rip)        # 0x51420e0 <writev+0x51420e0>
 367e156:      	movl	%edx, %esi
 367e158:      	movl	$0x5a, %edx
 367e15d:      	movq	%rax, %rdi
 367e160:      	xorl	%ecx, %ecx
 367e162:      	callq	*0x1ac3f80(%rip)        # 0x51420e8 <writev+0x51420e8>
 367e168:      	movq	%rax, 0x48(%rsp)
 367e16d:      	movl	%edx, 0x50(%rsp)
 367e171:      	movq	$0x15f90, 0x40(%rsp)    # imm = 0x15F90
 367e17a:      	movl	0x1acffb0(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451+0x8>
 367e180:      	testl	%eax, %eax
 367e182:      	jne	0x367e466 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4b6>
 367e188:      	movq	0x1acff99(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 367e18f:      	leaq	0x1ad5522(%rip), %r15   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 367e196:      	leaq	0x1ad5503(%rip), %r14   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 367e19d:      	leaq	0x108(%rsp), %rdi
 367e1a5:      	leaq	0x198(%rsp), %rsi
 367e1ad:      	movq	%rsp, %rcx
 367e1b0:      	leaq	0x40(%rsp), %r8
 367e1b5:      	xorl	%edx, %edx
 367e1b7:      	pushq	%r15
 367e1b9:      	pushq	%r14
 367e1bb:      	callq	0x3ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 367e1c0:      	addq	$0x10, %rsp
 367e1c4:      	cmpl	$-0x1, 0x108(%rsp)
 367e1cc:      	jne	0x367e477 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4c7>
 367e1d2:      	movq	0x140(%rsp), %rax
 367e1da:      	movq	%rax, 0x70(%rsp)
 367e1df:      	movups	0x110(%rsp), %xmm0
 367e1e7:      	movups	0x120(%rsp), %xmm1
 367e1ef:      	movups	0x130(%rsp), %xmm2
 367e1f7:      	movaps	%xmm2, 0x60(%rsp)
 367e1fc:      	movaps	%xmm1, 0x50(%rsp)
 367e201:      	movaps	%xmm0, 0x40(%rsp)
 367e206:      	leaq	0x40(%rsp), %rdi
 367e20b:      	callq	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 367e210:      	leaq	0x108(%rsp), %rdi
 367e218:      	leaq	0x198(%rsp), %rsi
 367e220:      	movq	%rsp, %rdx
 367e223:      	movl	$0x3e8, %ecx            # imm = 0x3E8
 367e228:      	callq	0x366f700 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention>
 367e22d:      	cmpl	$-0x1, 0x108(%rsp)
 367e235:      	jne	0x367e4b7 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x507>
 367e23b:      	leaq	0x40(%rsp), %rdi
 367e240:      	leaq	0x108(%rsp), %rsi
 367e248:      	movl	$0x90, %edx
 367e24d:      	callq	*0x1ac2ae5(%rip)        # 0x5140d38 <writev+0x5140d38>
 367e253:      	cmpl	$-0x1, 0x40(%rsp)
 367e258:      	je	0x367e269 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x2b9>
 367e25a:      	leaq	0x108(%rsp), %rdi
 367e262:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 367e267:      	jmp	0x367e273 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x2c3>
 367e269:      	leaq	0x48(%rsp), %rdi
 367e26e:      	callq	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 367e273:      	movl	0x1acff0f(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x18>
 367e279:      	testl	%eax, %eax
 367e27b:      	jne	0x367e4d2 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x522>
 367e281:      	movq	0x1acfef8(%rip), %rax   # 0x514e180 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x10>
 367e288:      	movq	%rax, 0xe8(%rsp)
 367e290:      	movq	$0x0, 0xf0(%rsp)
 367e29c:      	movq	$0x1, 0xf8(%rsp)
 367e2a8:      	movq	$0x0, 0x100(%rsp)
 367e2b4:      	callq	*0x1ac3e26(%rip)        # 0x51420e0 <writev+0x51420e0>
 367e2ba:      	movl	%edx, %esi
 367e2bc:      	movl	$0x5a, %edx
 367e2c1:      	movq	%rax, %rdi
 367e2c4:      	xorl	%ecx, %ecx
 367e2c6:      	callq	*0x1ac3e1c(%rip)        # 0x51420e8 <writev+0x51420e8>
 367e2cc:      	movq	%rax, 0xd8(%rsp)
 367e2d4:      	movl	%edx, 0xe0(%rsp)
 367e2db:      	movq	$0x15f90, 0xd0(%rsp)    # imm = 0x15F90
 367e2e7:      	movl	0x1acfe43(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451+0x8>
 367e2ed:      	testl	%eax, %eax
 367e2ef:      	jne	0x367e4e3 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x533>
 367e2f5:      	movq	0x1acfe2c(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 367e2fc:      	leaq	0x40(%rsp), %rdi
 367e301:      	leaq	0x198(%rsp), %rsi
 367e309:      	leaq	0xe8(%rsp), %rcx
 367e311:      	leaq	0xd0(%rsp), %r8
 367e319:      	movl	$0x2, %edx
 367e31e:      	pushq	%r15
 367e320:      	pushq	%r14
 367e322:      	callq	0x3ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 367e327:      	addq	$0x10, %rsp
 367e32b:      	cmpl	$-0x1, 0x40(%rsp)
 367e330:      	je	0x367e4f4 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x544>
 367e336:      	cmpq	$0xa, 0xc8(%rsp)
 367e33f:      	jne	0x367e4f4 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x544>
 367e345:      	movq	0xc0(%rsp), %rax
 367e34d:      	movabsq	$0x6575715f64656873, %rcx # imm = 0x6575715F64656873
 367e357:      	xorq	(%rax), %rcx
 367e35a:      	movzwl	0x8(%rax), %eax
 367e35e:      	xorq	$0x6575, %rax           # imm = 0x6575
 367e364:      	orq	%rcx, %rax
 367e367:      	jne	0x367e4f4 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x544>
 367e36d:      	leaq	0x40(%rsp), %rdi
 367e372:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 367e377:      	movq	0x18(%rsp), %rax
 367e37c:      	testq	%rax, %rax
 367e37f:      	je	0x367e38f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x3df>
 367e381:      	lock
 367e382:      	decq	(%rax)
 367e385:      	jne	0x367e38f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x3df>
 367e387:      	movq	%rbx, %rdi
 367e38a:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 367e38f:      	movq	$0x0, 0x1ad5316(%rip)   # 0x51536b0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS+0x10>
 367e39a:      	leaq	0x198(%rsp), %rdi
 367e3a2:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475>
 367e3a7:      	leaq	0x20(%rsp), %rdi
 367e3ac:      	callq	0x354e6d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server5tests22AdmissionCountersGuardEBF_>
 367e3b1:      	addq	$0x260, %rsp            # imm = 0x260
 367e3b8:      	popq	%rbx
 367e3b9:      	popq	%r14
 367e3bb:      	popq	%r15
 367e3bd:      	retq
 367e3be:      	leaq	0x1ad52fb(%rip), %rdi   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 367e3c5:      	callq	*0x1ac303d(%rip)        # 0x5141408 <writev+0x5141408>
 367e3cb:      	jmp	0x367dfd1 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x21>
 367e3d0:      	callq	*0x1ac2ca2(%rip)        # 0x5141078 <writev+0x5141078>
 367e3d6:      	movl	%eax, %ebx
 367e3d8:      	xorb	$0x1, %bl
 367e3db:      	movzbl	0x1ad52e2(%rip), %eax   # 0x51536c4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK+0x4>
 367e3e2:      	testb	%al, %al
 367e3e4:      	jne	0x367dff4 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x44>
 367e3ea:      	jmp	0x367dffb <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4b>
 367e3ef:      	leaq	0x1ad52ea(%rip), %rdi   # 0x51536e0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS>
 367e3f6:      	callq	*0x1ac300c(%rip)        # 0x5141408 <writev+0x5141408>
 367e3fc:      	jmp	0x367e018 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x68>
 367e401:      	callq	*0x1ac2c71(%rip)        # 0x5141078 <writev+0x5141078>
 367e407:      	xorb	$0x1, %al
 367e409:      	jmp	0x367e030 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x80>
 367e40e:      	callq	*0x1ac2d84(%rip)        # 0x5141198 <writev+0x5141198>
 367e414:      	movq	%rax, %fs:(%rbx)
 367e418:      	movq	%rdx, %fs:0x8(%rbx)
 367e41d:      	movb	$0x1, %fs:0x10(%rbx)
 367e422:      	jmp	0x367e072 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0xc2>
 367e427:      	leaq	0x1acfd42(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 367e42e:      	callq	0x3b2b248 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 367e433:      	jmp	0x367e0d4 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x124>
 367e438:      	leaq	0x1acfc91(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 367e43f:      	callq	0x3b2b3a1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 367e444:      	jmp	0x367e0e9 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x139>
 367e449:      	movq	$-0x1, %rbx
 367e450:      	jmp	0x367e108 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x158>
 367e455:      	leaq	0x1acfd14(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 367e45c:      	callq	0x3b2b248 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 367e461:      	jmp	0x367e12f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x17f>
 367e466:      	leaq	0x1acfcbb(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 367e46d:      	callq	0x3b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 367e472:      	jmp	0x367e188 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x1d8>
 367e477:      	leaq	0x40(%rsp), %r14
 367e47c:      	leaq	0x108(%rsp), %rsi
 367e484:      	movl	$0x90, %edx
 367e489:      	movq	%r14, %rdi
 367e48c:      	callq	*0x1ac28a6(%rip)        # 0x5140d38 <writev+0x5140d38>
 367e492:      	leaq	-0x3277029(%rip), %rdi  # 0x407470 <anon.60c0d423c0cc6e2470573bc6d977e900.3771.llvm.14966571587344355475+0x254>
 367e499:      	leaq	0x1971288(%rip), %rcx   # 0x4fef728 <anon.60c0d423c0cc6e2470573bc6d977e900.3562.llvm.14966571587344355475+0x1a0>
 367e4a0:      	leaq	0x1972109(%rip), %r8    # 0x4ff05b0 <anon.60c0d423c0cc6e2470573bc6d977e900.3878.llvm.14966571587344355475+0x18>
 367e4a7:      	movl	$0x3a, %esi
 367e4ac:      	movq	%r14, %rdx
 367e4af:      	callq	*0x1ac29e3(%rip)        # 0x5140e98 <writev+0x5140e98>
 367e4b5:      	jmp	0x367e517 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x567>
 367e4b7:      	leaq	-0x3276f7d(%rip), %rdi  # 0x407541 <anon.60c0d423c0cc6e2470573bc6d977e900.3771.llvm.14966571587344355475+0x325>
 367e4be:      	leaq	0x197211b(%rip), %rdx   # 0x4ff05e0 <anon.60c0d423c0cc6e2470573bc6d977e900.3878.llvm.14966571587344355475+0x48>
 367e4c5:      	movl	$0x139, %esi            # imm = 0x139
 367e4ca:      	callq	*0x1ac29b0(%rip)        # 0x5140e80 <writev+0x5140e80>
 367e4d0:      	jmp	0x367e517 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x567>
 367e4d2:      	leaq	0x1acfc97(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 367e4d9:      	callq	0x3b2b248 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 367e4de:      	jmp	0x367e281 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x2d1>
 367e4e3:      	leaq	0x1acfc3e(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 367e4ea:      	callq	0x3b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 367e4ef:      	jmp	0x367e2f5 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x345>
 367e4f4:      	leaq	0x40(%rsp), %rdi
 367e4f9:      	callq	0x352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>
 367e4fe:      	leaq	-0x327705b(%rip), %rdi  # 0x4074aa <anon.60c0d423c0cc6e2470573bc6d977e900.3771.llvm.14966571587344355475+0x28e>
 367e505:      	leaq	0x19720bc(%rip), %rdx   # 0x4ff05c8 <anon.60c0d423c0cc6e2470573bc6d977e900.3878.llvm.14966571587344355475+0x30>
 367e50c:      	movl	$0x97, %esi
 367e511:      	callq	*0x1ac29f1(%rip)        # 0x5140f08 <writev+0x5140f08>
 367e517:      	ud2
 367e519:      	movq	%rax, %r14
 367e51c:      	movzbl	%bl, %esi
 367e51f:      	leaq	0x1ad519a(%rip), %rdi   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 367e526:      	callq	0x35312d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 367e52b:      	jmp	0x367e59f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5ef>
 367e52d:      	callq	*0x1ac2885(%rip)        # 0x5140db8 <writev+0x5140db8>
 367e533:      	movq	%rax, %r14
 367e536:      	jmp	0x367e57d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5cd>
 367e538:      	jmp	0x367e541 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x591>
 367e53a:      	movq	%rax, %r14
 367e53d:      	jmp	0x367e595 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5e5>
 367e53f:      	jmp	0x367e541 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x591>
 367e541:      	movq	%rax, %r14
 367e544:      	jmp	0x367e565 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5b5>
 367e546:      	movq	%rax, %r14
 367e549:      	leaq	0x108(%rsp), %rdi
 367e551:      	callq	0x352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>
 367e556:      	jmp	0x367e565 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5b5>
 367e558:      	movq	%rax, %r14
 367e55b:      	leaq	0x40(%rsp), %rdi
 367e560:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 367e565:      	movq	0x18(%rsp), %rax
 367e56a:      	testq	%rax, %rax
 367e56d:      	je	0x367e57d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5cd>
 367e56f:      	lock
 367e570:      	decq	(%rax)
 367e573:      	jne	0x367e57d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5cd>
 367e575:      	movq	%rbx, %rdi
 367e578:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 367e57d:      	movq	$0x0, 0x1ad5128(%rip)   # 0x51536b0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS+0x10>
 367e588:      	leaq	0x198(%rsp), %rdi
 367e590:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475>
 367e595:      	leaq	0x20(%rsp), %rdi
 367e59a:      	callq	0x354e6d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server5tests22AdmissionCountersGuardEBF_>
 367e59f:      	movq	%r14, %rdi
 367e5a2:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 367e5a7:      	callq	*0x1ac280b(%rip)        # 0x5140db8 <writev+0x5140db8>
 367e5ad:      	callq	*0x1ac2805(%rip)        # 0x5140db8 <writev+0x5140db8>
 367e5b3:      	int3
 367e5b4:      	int3
 367e5b5:      	int3
 367e5b6:      	int3
 367e5b7:      	int3
 367e5b8:      	int3
 367e5b9:      	int3
 367e5ba:      	int3
 367e5bb:      	int3
 367e5bc:      	int3
 367e5bd:      	int3
 367e5be:      	int3
 367e5bf:      	int3

0000000003710430 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop>:
 3710430:      	pushq	%rbp
 3710431:      	pushq	%r15
 3710433:      	pushq	%r14
 3710435:      	pushq	%r13
 3710437:      	pushq	%r12
 3710439:      	pushq	%rbx
 371043a:      	subq	$0x278, %rsp            # imm = 0x278
 3710441:      	movl	$0x1, %ecx
 3710446:      	xorl	%eax, %eax
 3710448:      	lock
 3710449:      	cmpxchgl	%ecx, 0x1a43270(%rip)   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 3710450:      	jne	0x371082f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x3ff>
 3710456:      	movq	0x1a30c0b(%rip), %rbx   # 0x5141068 <writev+0x5141068>
 371045d:      	movq	(%rbx), %rax
 3710460:      	shlq	%rax
 3710463:      	testq	%rax, %rax
 3710466:      	jne	0x3710841 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x411>
 371046c:      	xorl	%ebp, %ebp
 371046e:      	movzbl	0x1a4324f(%rip), %eax   # 0x51536c4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK+0x4>
 3710475:      	testb	%al, %al
 3710477:      	je	0x3710480 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x50>
 3710479:      	movb	$0x0, 0x1a43244(%rip)   # 0x51536c4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK+0x4>
 3710480:      	xorl	%eax, %eax
 3710482:      	xchgb	%al, 0x1a43281(%rip)    # 0x5153709 <_RNvCs3pwlnhBXFtN_12memra_server8DRAINING>
 3710488:      	movl	$0x1, %ecx
 371048d:      	xorl	%eax, %eax
 371048f:      	lock
 3710490:      	cmpxchgl	%ecx, 0x1a43249(%rip)   # 0x51536e0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS>
 3710497:      	jne	0x3710861 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x431>
 371049d:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 37104a7:      	movq	(%rbx), %rcx
 37104aa:      	testq	%rax, %rcx
 37104ad:      	jne	0x3710873 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x443>
 37104b3:      	xorl	%eax, %eax
 37104b5:      	movzbl	0x1a43228(%rip), %ecx   # 0x51536e4 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS+0x4>
 37104bc:      	leaq	0x1a4321d(%rip), %rcx   # 0x51536e0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS>
 37104c3:      	movq	%rcx, 0x10(%rsp)
 37104c8:      	movb	%al, 0x18(%rsp)
 37104cc:      	leaq	0x1a431ed(%rip), %rax   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 37104d3:      	movq	%rax, 0x20(%rsp)
 37104d8:      	movb	%bpl, 0x28(%rsp)
 37104dd:      	movq	$-0x38, %rbx
 37104e4:      	cmpb	$0x1, %fs:0x10(%rbx)
 37104e9:      	jne	0x3710880 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x450>
 37104ef:      	movq	%fs:(%rbx), %rax
 37104f3:      	movq	%fs:0x8(%rbx), %rdx
 37104f8:      	leaq	0x1(%rax), %rcx
 37104fc:      	movq	%rcx, %fs:(%rbx)
 3710500:      	movups	0x18d4f31(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475>
 3710507:      	movaps	%xmm0, 0x50(%rsp)
 371050c:      	movups	0x18d4f35(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.14966571587344355475+0x10>
 3710513:      	movaps	%xmm0, 0x60(%rsp)
 3710518:      	movq	%rax, 0x70(%rsp)
 371051d:      	movq	%rdx, 0x78(%rsp)
 3710522:      	subq	$0x10, %rsp
 3710526:      	leaq	0x1c0(%rsp), %rdi
 371052e:      	leaq	0x60(%rsp), %r8
 3710533:      	movl	$0x1, %esi
 3710538:      	xorl	%edx, %edx
 371053a:      	xorl	%ecx, %ecx
 371053c:      	movq	$-0x1, %r9
 3710543:      	callq	0x366d170 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.14966571587344355475>
 3710548:      	addq	$0x10, %rsp
 371054c:      	movl	0x1a3dc36(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x18>
 3710552:      	testl	%eax, %eax
 3710554:      	jne	0x3710899 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x469>
 371055a:      	movq	0x1a3dc0f(%rip), %rbx   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 3710561:      	movl	0x1a3db79(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x10>
 3710567:      	testl	%eax, %eax
 3710569:      	jne	0x37108aa <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x47a>
 371056f:      	movq	0x1a3db5a(%rip), %rcx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 3710576:      	movq	0x1a3db5b(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x8>
 371057d:      	movq	%rbx, %rdx
 3710580:      	shrq	$0x3e, %rdx
 3710584:      	jne	0x37108bb <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x48b>
 371058a:      	leaq	(,%rbx,4), %r15
 3710592:      	testq	%rcx, %rcx
 3710595:      	cmovneq	%rax, %r15
 3710599:      	movq	%r15, 0x8(%rsp)
 371059e:      	testq	%r15, %r15
 37105a1:      	je	0x37108c7 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x497>
 37105a7:      	movq	%rbx, 0x30(%rsp)
 37105ac:      	movq	$0x0, 0x38(%rsp)
 37105b5:      	movq	$0x1, 0x40(%rsp)
 37105be:      	movq	$0x0, 0x48(%rsp)
 37105c7:      	movq	0x1a430ea(%rip), %rax   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 37105ce:      	xorl	%ecx, %ecx
 37105d0:      	lock
 37105d1:      	cmpxchgq	%rcx, 0x1a430df(%rip)   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 37105d9:      	jne	0x37105d0 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x1a0>
 37105db:      	leaq	-0x1(%r15), %rbp
 37105df:      	movq	%rbp, 0x1a430ba(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 37105e6:      	callq	*0x1a31af4(%rip)        # 0x51420e0 <writev+0x51420e0>
 37105ec:      	movl	%edx, %esi
 37105ee:      	movl	$0x5a, %edx
 37105f3:      	movq	%rax, %rdi
 37105f6:      	xorl	%ecx, %ecx
 37105f8:      	callq	*0x1a31aea(%rip)        # 0x51420e8 <writev+0x51420e8>
 37105fe:      	movq	%rax, 0x58(%rsp)
 3710603:      	movl	%edx, 0x60(%rsp)
 3710607:      	movq	$0x15f90, 0x50(%rsp)    # imm = 0x15F90
 3710610:      	movl	0x1a3db1a(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451+0x8>
 3710616:      	testl	%eax, %eax
 3710618:      	jne	0x37108e5 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x4b5>
 371061e:      	movq	0x1a3db03(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 3710625:      	leaq	0x1a4308c(%rip), %r13   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 371062c:      	leaq	0x1a4306d(%rip), %r12   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 3710633:      	leaq	0x120(%rsp), %rbx
 371063b:      	leaq	0x1b0(%rsp), %rsi
 3710643:      	leaq	0x30(%rsp), %rcx
 3710648:      	leaq	0x50(%rsp), %r14
 371064d:      	movq	%rbx, %rdi
 3710650:      	xorl	%edx, %edx
 3710652:      	movq	%r14, %r8
 3710655:      	pushq	%r13
 3710657:      	pushq	%r12
 3710659:      	callq	0x3ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 371065e:      	addq	$0x10, %rsp
 3710662:      	cmpl	$-0x1, 0x120(%rsp)
 371066a:      	jne	0x37108f6 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x4c6>
 3710670:      	movq	0x158(%rsp), %rax
 3710678:      	movq	%rax, 0x110(%rsp)
 3710680:      	movups	0x128(%rsp), %xmm0
 3710688:      	movups	0x138(%rsp), %xmm1
 3710690:      	movups	0x148(%rsp), %xmm2
 3710698:      	movaps	%xmm2, 0x100(%rsp)
 37106a0:      	movaps	%xmm1, 0xf0(%rsp)
 37106a8:      	movaps	%xmm0, 0xe0(%rsp)
 37106b0:      	movq	0x1a43001(%rip), %rax   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 37106b7:      	movq	%rax, 0x50(%rsp)
 37106bc:      	cmpq	$0x1, %rax
 37106c0:      	jne	0x3710936 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x506>
 37106c6:      	movq	0x1a42fd3(%rip), %rax   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 37106cd:      	movq	%rax, 0x50(%rsp)
 37106d2:      	cmpq	%r15, %rax
 37106d5:      	jne	0x3710946 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x516>
 37106db:      	leaq	0xe0(%rsp), %rdi
 37106e3:      	callq	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 37106e8:      	movq	0x1a42fc9(%rip), %rax   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 37106ef:      	movq	%rax, 0x50(%rsp)
 37106f4:      	testq	%rax, %rax
 37106f7:      	jne	0x3710961 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x531>
 37106fd:      	movq	0x1a42f9c(%rip), %rax   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 3710704:      	movq	%rax, 0x120(%rsp)
 371070c:      	movq	%rbp, 0x50(%rsp)
 3710711:      	cmpq	%rbp, %rax
 3710714:      	jne	0x3710976 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x546>
 371071a:      	movq	%r15, 0x1a42f7f(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 3710721:      	callq	*0x1a319b9(%rip)        # 0x51420e0 <writev+0x51420e0>
 3710727:      	movl	%edx, %esi
 3710729:      	movl	$0x5a, %edx
 371072e:      	movq	%rax, %rdi
 3710731:      	xorl	%ecx, %ecx
 3710733:      	callq	*0x1a319af(%rip)        # 0x51420e8 <writev+0x51420e8>
 3710739:      	movq	%rax, 0x128(%rsp)
 3710741:      	movl	%edx, 0x130(%rsp)
 3710748:      	movq	$0x15f90, 0x120(%rsp)   # imm = 0x15F90
 3710754:      	movl	0x1a3d9d6(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451+0x8>
 371075a:      	testl	%eax, %eax
 371075c:      	jne	0x3710991 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x561>
 3710762:      	movq	0x1a3d9bf(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 3710769:      	leaq	0x50(%rsp), %rdi
 371076e:      	leaq	0x1b0(%rsp), %rsi
 3710776:      	leaq	0x30(%rsp), %rcx
 371077b:      	leaq	0x120(%rsp), %r8
 3710783:      	xorl	%edx, %edx
 3710785:      	pushq	%r13
 3710787:      	pushq	%r12
 3710789:      	callq	0x3ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 371078e:      	addq	$0x10, %rsp
 3710792:      	cmpq	$0xa, 0xd8(%rsp)
 371079b:      	sete	%al
 371079e:      	cmpl	$-0x1, 0x50(%rsp)
 37107a3:      	je	0x371080b <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x3db>
 37107a5:      	testb	%al, %al
 37107a7:      	je	0x3710811 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x3e1>
 37107a9:      	movq	0xd0(%rsp), %rax
 37107b1:      	movabsq	$0x6575715f64656873, %rcx # imm = 0x6575715F64656873
 37107bb:      	xorq	(%rax), %rcx
 37107be:      	movzwl	0x8(%rax), %eax
 37107c2:      	xorq	$0x6575, %rax           # imm = 0x6575
 37107c8:      	orq	%rcx, %rax
 37107cb:      	jne	0x3710811 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x3e1>
 37107cd:      	leaq	0x50(%rsp), %rdi
 37107d2:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 37107d7:      	movq	$0x0, 0x1a42ebe(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 37107e2:      	leaq	0x1b0(%rsp), %rdi
 37107ea:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475>
 37107ef:      	leaq	0x10(%rsp), %rdi
 37107f4:      	callq	0x354e6d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server5tests22AdmissionCountersGuardEBF_>
 37107f9:      	addq	$0x278, %rsp            # imm = 0x278
 3710800:      	popq	%rbx
 3710801:      	popq	%r12
 3710803:      	popq	%r13
 3710805:      	popq	%r14
 3710807:      	popq	%r15
 3710809:      	popq	%rbp
 371080a:      	retq
 371080b:      	xorl	%eax, %eax
 371080d:      	testb	%al, %al
 371080f:      	jne	0x37107a9 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x379>
 3710811:      	leaq	-0x31b4cb5(%rip), %rdi  # 0x55bb63 <anon.60c0d423c0cc6e2470573bc6d977e900.6219.llvm.14966571587344355475+0x2625>
 3710818:      	leaq	0x18e7331(%rip), %rdx   # 0x4ff7b50 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x1b0>
 371081f:      	movl	$0x3c, %esi
 3710824:      	callq	*0x1a306de(%rip)        # 0x5140f08 <writev+0x5140f08>
 371082a:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 371082f:      	leaq	0x1a42e8a(%rip), %rdi   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 3710836:      	callq	*0x1a30bcc(%rip)        # 0x5141408 <writev+0x5141408>
 371083c:      	jmp	0x3710456 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x26>
 3710841:      	callq	*0x1a30831(%rip)        # 0x5141078 <writev+0x5141078>
 3710847:      	movl	%eax, %ebp
 3710849:      	xorb	$0x1, %bpl
 371084d:      	movzbl	0x1a42e70(%rip), %eax   # 0x51536c4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK+0x4>
 3710854:      	testb	%al, %al
 3710856:      	jne	0x3710479 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x49>
 371085c:      	jmp	0x3710480 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x50>
 3710861:      	leaq	0x1a42e78(%rip), %rdi   # 0x51536e0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server5tests23admission_counters_lock8COUNTERS>
 3710868:      	callq	*0x1a30b9a(%rip)        # 0x5141408 <writev+0x5141408>
 371086e:      	jmp	0x371049d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x6d>
 3710873:      	callq	*0x1a307ff(%rip)        # 0x5141078 <writev+0x5141078>
 3710879:      	xorb	$0x1, %al
 371087b:      	jmp	0x37104b5 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x85>
 3710880:      	callq	*0x1a30912(%rip)        # 0x5141198 <writev+0x5141198>
 3710886:      	movq	%rax, %fs:(%rbx)
 371088a:      	movq	%rdx, %fs:0x8(%rbx)
 371088f:      	movb	$0x1, %fs:0x10(%rbx)
 3710894:      	jmp	0x37104f8 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0xc8>
 3710899:      	leaq	0x1a3d8d0(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 37108a0:      	callq	0x3b2b248 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 37108a5:      	jmp	0x371055a <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x12a>
 37108aa:      	leaq	0x1a3d81f(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 37108b1:      	callq	0x3b2b3a1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 37108b6:      	jmp	0x371056f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x13f>
 37108bb:      	movq	$-0x1, %r15
 37108c2:      	jmp	0x3710592 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x162>
 37108c7:      	leaq	-0x31b4dc3(%rip), %rdi  # 0x55bb0b <anon.60c0d423c0cc6e2470573bc6d977e900.6219.llvm.14966571587344355475+0x25cd>
 37108ce:      	leaq	0x18e721b(%rip), %rdx   # 0x4ff7af0 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x150>
 37108d5:      	movl	$0x5f, %esi
 37108da:      	callq	*0x1a305a0(%rip)        # 0x5140e80 <writev+0x5140e80>
 37108e0:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 37108e5:      	leaq	0x1a3d83c(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 37108ec:      	callq	0x3b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 37108f1:      	jmp	0x371061e <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x1ee>
 37108f6:      	leaq	0x50(%rsp), %rbx
 37108fb:      	leaq	0x120(%rsp), %rsi
 3710903:      	movl	$0x90, %edx
 3710908:      	movq	%rbx, %rdi
 371090b:      	callq	*0x1a30427(%rip)        # 0x5140d38 <writev+0x5140d38>
 3710911:      	leaq	-0x31b4dde(%rip), %rdi  # 0x55bb3a <anon.60c0d423c0cc6e2470573bc6d977e900.6219.llvm.14966571587344355475+0x25fc>
 3710918:      	leaq	0x18dee09(%rip), %rcx   # 0x4fef728 <anon.60c0d423c0cc6e2470573bc6d977e900.3562.llvm.14966571587344355475+0x1a0>
 371091f:      	leaq	0x18e71e2(%rip), %r8    # 0x4ff7b08 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x168>
 3710926:      	movl	$0x29, %esi
 371092b:      	movq	%rbx, %rdx
 371092e:      	callq	*0x1a30564(%rip)        # 0x5140e98 <writev+0x5140e98>
 3710934:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 3710936:      	leaq	0x18e7243(%rip), %r9    # 0x4ff7b80 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x1e0>
 371093d:      	leaq	-0x33866e4(%rip), %rdx  # 0x38a260 <anon.ea9ff54ec891c523b22d7beb433c6217.3025.llvm.10671892735646583555>
 3710944:      	jmp	0x3710952 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x522>
 3710946:      	leaq	0x18e71d3(%rip), %r9    # 0x4ff7b20 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x180>
 371094d:      	leaq	0x8(%rsp), %rdx
 3710952:      	xorl	%edi, %edi
 3710954:      	movq	%r14, %rsi
 3710957:      	xorl	%ecx, %ecx
 3710959:      	callq	*0x1a30671(%rip)        # 0x5140fd0 <writev+0x5140fd0>
 371095f:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 3710961:      	leaq	0x18e7200(%rip), %r9    # 0x4ff7b68 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x1c8>
 3710968:      	leaq	-0x33864d7(%rip), %rdx  # 0x38a498 <anon.c030ccd08983dd522ef199ffd40d4bd6.616.llvm.15771381525616154697>
 371096f:      	leaq	0x50(%rsp), %rbx
 3710974:      	jmp	0x3710982 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x552>
 3710976:      	leaq	0x18e71bb(%rip), %r9    # 0x4ff7b38 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.14966571587344355475+0x198>
 371097d:      	leaq	0x50(%rsp), %rdx
 3710982:      	xorl	%edi, %edi
 3710984:      	movq	%rbx, %rsi
 3710987:      	xorl	%ecx, %ecx
 3710989:      	callq	*0x1a30641(%rip)        # 0x5140fd0 <writev+0x5140fd0>
 371098f:      	ud2
 3710991:      	leaq	0x1a3d790(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.15708504647765068451>
 3710998:      	callq	0x3b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 371099d:      	jmp	0x3710762 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x332>
 37109a2:      	movq	%rax, %rbx
 37109a5:      	movzbl	%bpl, %esi
 37109a9:      	leaq	0x1a42d10(%rip), %rdi   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 37109b0:      	callq	0x35312d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 37109b5:      	jmp	0x3710a24 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5f4>
 37109b7:      	callq	*0x1a303fb(%rip)        # 0x5140db8 <writev+0x5140db8>
 37109bd:      	movq	%rax, %rbx
 37109c0:      	jmp	0x3710a1a <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5ea>
 37109c2:      	jmp	0x37109f0 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5c0>
 37109c4:      	movq	%rax, %rbx
 37109c7:      	leaq	0x50(%rsp), %rdi
 37109cc:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.14966571587344355475>
 37109d1:      	jmp	0x3710a02 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5d2>
 37109d3:      	callq	*0x1a303df(%rip)        # 0x5140db8 <writev+0x5140db8>
 37109d9:      	movq	%rax, %rbx
 37109dc:      	jmp	0x3710a0d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5dd>
 37109de:      	movq	%rax, %rbx
 37109e1:      	leaq	0xe0(%rsp), %rdi
 37109e9:      	callq	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 37109ee:      	jmp	0x3710a02 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5d2>
 37109f0:      	movq	%rax, %rbx
 37109f3:      	jmp	0x3710a02 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5d2>
 37109f5:      	movq	%rax, %rbx
 37109f8:      	leaq	0x50(%rsp), %rdi
 37109fd:      	callq	0x352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>
 3710a02:      	movq	$0x0, 0x1a42c93(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 3710a0d:      	leaq	0x1b0(%rsp), %rdi
 3710a15:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.14966571587344355475>
 3710a1a:      	leaq	0x10(%rsp), %rdi
 3710a1f:      	callq	0x354e6d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server5tests22AdmissionCountersGuardEBF_>
 3710a24:      	movq	%rbx, %rdi
 3710a27:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3710a2c:      	callq	*0x1a30386(%rip)        # 0x5140db8 <writev+0x5140db8>
 3710a32:      	int3
 3710a33:      	int3
 3710a34:      	int3
 3710a35:      	int3
 3710a36:      	int3
 3710a37:      	int3
 3710a38:      	int3
 3710a39:      	int3
 3710a3a:      	int3
 3710a3b:      	int3
 3710a3c:      	int3
 3710a3d:      	int3
 3710a3e:      	int3
 3710a3f:      	int3

000000000384d580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_>:
 384d580:      	pushq	%rbx
 384d581:      	movq	%rdi, %rbx
 384d584:      	movq	0x10(%rdi), %rsi
 384d588:      	testq	%rsi, %rsi
 384d58b:      	je	0x384d59c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_+0x1c>
 384d58d:      	movq	0x18(%rbx), %rdi
 384d591:      	movl	$0x1, %edx
 384d596:      	callq	*0x18f37a4(%rip)        # 0x5140d40 <writev+0x5140d40>
 384d59c:      	movq	0x28(%rbx), %rsi
 384d5a0:      	cmpq	$-0x1, %rsi
 384d5a4:      	je	0x384d5bb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_+0x3b>
 384d5a6:      	testq	%rsi, %rsi
 384d5a9:      	je	0x384d5bb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_+0x3b>
 384d5ab:      	movq	0x30(%rbx), %rdi
 384d5af:      	movl	$0x1, %edx
 384d5b4:      	popq	%rbx
 384d5b5:      	jmpq	*0x18f3785(%rip)        # 0x5140d40 <writev+0x5140d40>
 384d5bb:      	popq	%rbx
 384d5bc:      	retq
 384d5bd:      	int3
 384d5be:      	int3
 384d5bf:      	int3

0000000003875400 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms>:
 3875400:      	pushq	%r14
 3875402:      	pushq	%rbx
 3875403:      	subq	$0x228, %rsp            # imm = 0x228
 387540a:      	movq	%rdi, %rbx
 387540d:      	leaq	0x68(%rsp), %rdi
 3875412:      	callq	0x388f880 <_RNvXs1_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealthNtNtCs4NRVxsYgnAr_4core7default7Default7default>
 3875417:      	movl	0x138(%rsp), %eax
 387541e:      	movups	0x68(%rsp), %xmm0
 3875423:      	movups	0x78(%rsp), %xmm1
 3875428:      	movups	0x88(%rsp), %xmm2
 3875430:      	movups	0x98(%rsp), %xmm3
 3875438:      	movaps	%xmm0, (%rsp)
 387543c:      	movaps	%xmm1, 0x10(%rsp)
 3875441:      	movaps	%xmm2, 0x20(%rsp)
 3875446:      	movaps	%xmm3, 0x30(%rsp)
 387544b:      	movups	0xa8(%rsp), %xmm0
 3875453:      	movaps	%xmm0, 0x40(%rsp)
 3875458:      	movups	0xb8(%rsp), %xmm0
 3875460:      	movaps	%xmm0, 0x50(%rsp)
 3875465:      	movq	0x130(%rsp), %rcx
 387546d:      	movq	%rcx, 0x218(%rsp)
 3875475:      	movups	0x120(%rsp), %xmm0
 387547d:      	movups	%xmm0, 0x208(%rsp)
 3875485:      	movups	0x110(%rsp), %xmm0
 387548d:      	movups	%xmm0, 0x1f8(%rsp)
 3875495:      	movq	$0x1, 0x140(%rsp)
 38754a1:      	movq	$0x1, 0x148(%rsp)
 38754ad:      	movaps	(%rsp), %xmm0
 38754b1:      	movaps	0x10(%rsp), %xmm1
 38754b6:      	movaps	0x20(%rsp), %xmm2
 38754bb:      	movaps	0x30(%rsp), %xmm3
 38754c0:      	movups	%xmm0, 0x150(%rsp)
 38754c8:      	movups	%xmm1, 0x160(%rsp)
 38754d0:      	movups	%xmm2, 0x170(%rsp)
 38754d8:      	movups	%xmm3, 0x180(%rsp)
 38754e0:      	movaps	0x40(%rsp), %xmm0
 38754e5:      	movups	%xmm0, 0x190(%rsp)
 38754ed:      	movaps	0x50(%rsp), %xmm0
 38754f2:      	movups	%xmm0, 0x1a0(%rsp)
 38754fa:      	movq	$0x0, 0x1b0(%rsp)
 3875506:      	movups	0xd8(%rsp), %xmm0
 387550e:      	movups	%xmm0, 0x1c0(%rsp)
 3875516:      	movups	0xe8(%rsp), %xmm0
 387551e:      	movups	%xmm0, 0x1d0(%rsp)
 3875526:      	movups	0xf8(%rsp), %xmm0
 387552e:      	movups	%xmm0, 0x1e0(%rsp)
 3875536:      	movq	%rbx, 0x1f0(%rsp)
 387553e:      	movl	%eax, 0x220(%rsp)
 3875545:      	movl	0x13c(%rsp), %eax
 387554c:      	movl	%eax, 0x224(%rsp)
 3875553:      	callq	*0x18cb837(%rip)        # 0x5140d90 <writev+0x5140d90>
 3875559:      	movl	$0xe8, %edi
 387555e:      	movl	$0x8, %esi
 3875563:      	callq	*0x18cb82f(%rip)        # 0x5140d98 <writev+0x5140d98>
 3875569:      	testq	%rax, %rax
 387556c:      	je	0x38755b5 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1b5>
 387556e:      	movq	%rax, %rbx
 3875571:      	leaq	0x140(%rsp), %rsi
 3875579:      	movl	$0xe8, %edx
 387557e:      	movq	%rax, %rdi
 3875581:      	callq	*0x18cb7b1(%rip)        # 0x5140d38 <writev+0x5140d38>
 3875587:      	movq	0xc8(%rsp), %rax
 387558f:      	testq	%rax, %rax
 3875592:      	je	0x38755a7 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1a7>
 3875594:      	lock
 3875595:      	decq	(%rax)
 3875598:      	jne	0x38755a7 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1a7>
 387559a:      	leaq	0xc8(%rsp), %rdi
 38755a2:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 38755a7:      	movq	%rbx, %rax
 38755aa:      	addq	$0x228, %rsp            # imm = 0x228
 38755b1:      	popq	%rbx
 38755b2:      	popq	%r14
 38755b4:      	retq
 38755b5:      	leaq	0x150(%rsp), %r14
 38755bd:      	movl	$0x8, %edi
 38755c2:      	movl	$0xe8, %esi
 38755c7:      	callq	*0x18cb993(%rip)        # 0x5140f60 <writev+0x5140f60>
 38755cd:      	ud2
 38755cf:      	movq	%rax, %rbx
 38755d2:      	movq	%r14, %rdi
 38755d5:      	callq	0x384d3c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthEBF_.llvm.5209651971967471703>
 38755da:      	movq	0xc8(%rsp), %rax
 38755e2:      	testq	%rax, %rax
 38755e5:      	je	0x38755fa <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1fa>
 38755e7:      	lock
 38755e8:      	decq	(%rax)
 38755eb:      	jne	0x38755fa <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1fa>
 38755ed:      	leaq	0xc8(%rsp), %rdi
 38755f5:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 38755fa:      	movq	%rbx, %rdi
 38755fd:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3875602:      	callq	*0x18cb7b0(%rip)        # 0x5140db8 <writev+0x5140db8>
 3875608:      	callq	*0x18cb7aa(%rip)        # 0x5140db8 <writev+0x5140db8>
 387560e:      	int3
 387560f:      	int3

0000000003875610 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route>:
 3875610:      	pushq	%rbp
 3875611:      	pushq	%r15
 3875613:      	pushq	%r14
 3875615:      	pushq	%r13
 3875617:      	pushq	%r12
 3875619:      	pushq	%rbx
 387561a:      	subq	$0xc8, %rsp
 3875621:      	movq	%rdi, %rbx
 3875624:      	movq	%rsi, 0x18(%rsp)
 3875629:      	movq	%rdx, 0x20(%rsp)
 387562e:      	leaq	0x38(%rsp), %rdi
 3875633:      	callq	0x3878ac0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth3new>
 3875638:      	movq	$0x1, 0x28(%rsp)
 3875641:      	movq	$0x1, 0x30(%rsp)
 387564a:      	callq	*0x18cb740(%rip)        # 0x5140d90 <writev+0x5140d90>
 3875650:      	movl	$0xa0, %edi
 3875655:      	movl	$0x8, %esi
 387565a:      	callq	*0x18cb738(%rip)        # 0x5140d98 <writev+0x5140d98>
 3875660:      	testq	%rax, %rax
 3875663:      	je	0x3875774 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x164>
 3875669:      	movq	%rax, %r14
 387566c:      	leaq	0x28(%rsp), %rsi
 3875671:      	movl	$0xa0, %edx
 3875676:      	movq	%rax, %rdi
 3875679:      	callq	*0x18cb6b9(%rip)        # 0x5140d38 <writev+0x5140d38>
 387567f:      	movq	%r14, 0x8(%rsp)
 3875684:      	leaq	0xa8(%rbx), %r14
 387568b:      	movl	$0x3fffffff, %ecx       # imm = 0x3FFFFFFF
 3875690:      	xorl	%eax, %eax
 3875692:      	lock
 3875693:      	cmpxchgl	%ecx, 0xa8(%rbx)
 387569a:      	jne	0x3875786 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x176>
 38756a0:      	movq	0x18cb9c1(%rip), %rax   # 0x5141068 <writev+0x5141068>
 38756a7:      	movq	(%rax), %rax
 38756aa:      	shlq	%rax
 38756ad:      	testq	%rax, %rax
 38756b0:      	jne	0x3875794 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x184>
 38756b6:      	xorl	%ebp, %ebp
 38756b8:      	movzbl	0xb0(%rbx), %eax
 38756bf:      	movq	%r14, 0x28(%rsp)
 38756c4:      	movb	%bpl, 0x30(%rsp)
 38756c9:      	leaq	0xb8(%rbx), %r12
 38756d0:      	leaq	0x18(%rsp), %rsi
 38756d5:      	movq	%r12, %rdi
 38756d8:      	callq	0x3adc8a0 <_RINvMs_NtCscdodAO9FK5_5alloc3vecINtB5_3VecINtNtB7_4sync3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthEE6retainNCNvMs2_BX_NtBX_12WorkerHealth14register_routes_0EBZ_>
 38756dd:      	movq	0x8(%rsp), %r15
 38756e2:      	lock
 38756e3:      	incq	(%r15)
 38756e6:      	jle	0x3875784 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x174>
 38756ec:      	movq	%r15, 0x10(%rsp)
 38756f1:      	movq	0xc8(%rbx), %r13
 38756f8:      	cmpq	0xb8(%rbx), %r13
 38756ff:      	jne	0x3875709 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0xf9>
 3875701:      	movq	%r12, %rdi
 3875704:      	callq	0x39b3560 <_RNvMs3_NtCscdodAO9FK5_5alloc7raw_vecINtB5_6RawVecQNtCskf97wSN6mZ4_8memra_kv5CacheE8grow_oneCs3pwlnhBXFtN_12memra_server>
 3875709:      	movq	0xc0(%rbx), %rax
 3875710:      	movq	%r15, (%rax,%r13,8)
 3875714:      	incq	%r13
 3875717:      	movq	%r13, 0xc8(%rbx)
 387571e:      	testb	%bpl, %bpl
 3875721:      	jne	0x387573c <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x12c>
 3875723:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 387572d:      	movq	0x18cb934(%rip), %rcx   # 0x5141068 <writev+0x5141068>
 3875734:      	movq	(%rcx), %rcx
 3875737:      	testq	%rax, %rcx
 387573a:      	jne	0x38757a5 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x195>
 387573c:      	movl	$0xc0000001, %esi       # imm = 0xC0000001
 3875741:      	lock
 3875742:      	xaddl	%esi, (%r14)
 3875746:      	addl	$0xc0000001, %esi       # imm = 0xC0000001
 387574c:      	cmpl	$0x40000000, %esi       # imm = 0x40000000
 3875752:      	jae	0x3875769 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x159>
 3875754:      	movq	%r15, %rax
 3875757:      	addq	$0xc8, %rsp
 387575e:      	popq	%rbx
 387575f:      	popq	%r12
 3875761:      	popq	%r13
 3875763:      	popq	%r14
 3875765:      	popq	%r15
 3875767:      	popq	%rbp
 3875768:      	retq
 3875769:      	movq	%r14, %rdi
 387576c:      	callq	*0x18cc7d6(%rip)        # 0x5141f48 <writev+0x5141f48>
 3875772:      	jmp	0x3875754 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x144>
 3875774:      	movl	$0x8, %edi
 3875779:      	movl	$0xa0, %esi
 387577e:      	callq	*0x18cb7dc(%rip)        # 0x5140f60 <writev+0x5140f60>
 3875784:      	ud2
 3875786:      	movq	%r14, %rdi
 3875789:      	callq	*0x18cca89(%rip)        # 0x5142218 <writev+0x5142218>
 387578f:      	jmp	0x38756a0 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x90>
 3875794:      	callq	*0x18cb8de(%rip)        # 0x5141078 <writev+0x5141078>
 387579a:      	movl	%eax, %ebp
 387579c:      	xorb	$0x1, %bpl
 38757a0:      	jmp	0x38756b8 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0xa8>
 38757a5:      	callq	*0x18cb8cd(%rip)        # 0x5141078 <writev+0x5141078>
 38757ab:      	testb	%al, %al
 38757ad:      	jne	0x387573c <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x12c>
 38757af:      	movb	$0x1, 0xb0(%rbx)
 38757b6:      	jmp	0x387573c <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x12c>
 38757b8:      	movq	%rax, %rbx
 38757bb:      	jmp	0x38757e5 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1d5>
 38757bd:      	movq	%rax, %rbx
 38757c0:      	lock
 38757c1:      	decq	(%r15)
 38757c4:      	jne	0x38757db <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1cb>
 38757c6:      	leaq	0x10(%rsp), %rdi
 38757cb:      	callq	0x39bcef0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 38757d0:      	jmp	0x38757db <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1cb>
 38757d2:      	callq	*0x18cb5e0(%rip)        # 0x5140db8 <writev+0x5140db8>
 38757d8:      	movq	%rax, %rbx
 38757db:      	leaq	0x28(%rsp), %rdi
 38757e0:      	callq	0x3ba8990 <_RNvXsi_NtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlockINtB5_16RwLockWriteGuardNtNtCs3pwlnhBXFtN_12memra_server4auth5StateENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1e_>
 38757e5:      	movq	0x8(%rsp), %rax
 38757ea:      	lock
 38757eb:      	decq	(%rax)
 38757ee:      	jne	0x387580f <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1ff>
 38757f0:      	leaq	0x8(%rsp), %rdi
 38757f5:      	callq	0x39bcef0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 38757fa:      	jmp	0x387580f <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1ff>
 38757fc:      	callq	*0x18cb5b6(%rip)        # 0x5140db8 <writev+0x5140db8>
 3875802:      	movq	%rax, %rbx
 3875805:      	leaq	0x28(%rsp), %rdi
 387580a:      	callq	0x384c930 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCscdodAO9FK5_5alloc4sync8ArcInnerNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthEEB1h_>
 387580f:      	movq	%rbx, %rdi
 3875812:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3875817:      	callq	*0x18cb59b(%rip)        # 0x5140db8 <writev+0x5140db8>
 387581d:      	int3
 387581e:      	int3
 387581f:      	int3

0000000003878690 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>:
 3878690:      	pushq	%rbp
 3878691:      	pushq	%r15
 3878693:      	pushq	%r14
 3878695:      	pushq	%r12
 3878697:      	pushq	%rbx
 3878698:      	subq	$0x10, %rsp
 387869c:      	movq	%rdi, %r14
 387869f:      	leaq	0x60(%rdi), %rbx
 38786a3:      	movl	$0x1, %ecx
 38786a8:      	xorl	%eax, %eax
 38786aa:      	lock
 38786ab:      	cmpxchgl	%ecx, 0x60(%rdi)
 38786af:      	jne	0x387878b <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xfb>
 38786b5:      	movq	0x18c89ac(%rip), %r15   # 0x5141068 <writev+0x5141068>
 38786bc:      	movq	(%r15), %rax
 38786bf:      	shlq	%rax
 38786c2:      	testq	%rax, %rax
 38786c5:      	jne	0x3878799 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x109>
 38786cb:      	xorl	%ebp, %ebp
 38786cd:      	movabsq	$0x7fffffffffffffff, %r12 # imm = 0x7FFFFFFFFFFFFFFF
 38786d7:      	movzbl	0x64(%r14), %eax
 38786dc:      	movq	0x58(%r14), %rax
 38786e0:      	testq	%rax, %rax
 38786e3:      	je	0x38786f7 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x67>
 38786e5:      	leaq	-0x1(%rax), %rcx
 38786e9:      	lock
 38786ea:      	cmpxchgq	%rcx, 0x58(%r14)
 38786ef:      	jne	0x38786e0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x50>
 38786f1:      	cmpq	$0x1, %rax
 38786f5:      	ja	0x3878768 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xd8>
 38786f7:      	movzbl	0x88(%r14), %eax
 38786ff:      	cmpb	$0x3, %al
 3878701:      	je	0x3878768 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xd8>
 3878703:      	cmpb	$0x0, %fs:-0x1c90
 387870c:      	je	0x3878719 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x89>
 387870e:      	movq	%fs:-0x1c88, %rax
 3878717:      	jmp	0x387875c <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xcc>
 3878719:      	movl	0x18d57d1(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x10>
 387871f:      	testl	%eax, %eax
 3878721:      	jne	0x38787c6 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x136>
 3878727:      	movq	0x18d57b2(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 387872e:      	movl	0x18d57b4(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x8>
 3878734:      	movq	%rax, (%rsp)
 3878738:      	movl	%ecx, 0x8(%rsp)
 387873c:      	movq	%rsp, %rdi
 387873f:      	callq	*0x18c8773(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 3878745:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 387874c:      	movl	%edx, %eax
 387874e:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 3878755:      	shrq	$0x32, %rax
 3878759:      	addq	%rcx, %rax
 387875c:      	movq	%rax, 0x28(%r14)
 3878760:      	movb	$0x1, 0x88(%r14)
 3878768:      	testb	%bpl, %bpl
 387876b:      	jne	0x3878775 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xe5>
 387876d:      	movq	(%r15), %rax
 3878770:      	testq	%r12, %rax
 3878773:      	jne	0x38787b5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x125>
 3878775:      	xorl	%eax, %eax
 3878777:      	xchgl	%eax, (%rbx)
 3878779:      	cmpl	$0x2, %eax
 387877c:      	je	0x38787aa <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x11a>
 387877e:      	addq	$0x10, %rsp
 3878782:      	popq	%rbx
 3878783:      	popq	%r12
 3878785:      	popq	%r14
 3878787:      	popq	%r15
 3878789:      	popq	%rbp
 387878a:      	retq
 387878b:      	movq	%rbx, %rdi
 387878e:      	callq	*0x18c8c74(%rip)        # 0x5141408 <writev+0x5141408>
 3878794:      	jmp	0x38786b5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x25>
 3878799:      	callq	*0x18c88d9(%rip)        # 0x5141078 <writev+0x5141078>
 387879f:      	movl	%eax, %ebp
 38787a1:      	xorb	$0x1, %bpl
 38787a5:      	jmp	0x38786cd <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x3d>
 38787aa:      	movq	%rbx, %rdi
 38787ad:      	callq	*0x18c88bd(%rip)        # 0x5141070 <writev+0x5141070>
 38787b3:      	jmp	0x387877e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xee>
 38787b5:      	callq	*0x18c88bd(%rip)        # 0x5141078 <writev+0x5141078>
 38787bb:      	testb	%al, %al
 38787bd:      	jne	0x3878775 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xe5>
 38787bf:      	movb	$0x1, 0x64(%r14)
 38787c4:      	jmp	0x3878775 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xe5>
 38787c6:      	leaq	0x18d5713(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 38787cd:      	callq	0x3b2b87b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 38787d2:      	jmp	0x3878727 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x97>
 38787d7:      	movq	%rax, %r14
 38787da:      	movzbl	%bpl, %esi
 38787de:      	movq	%rbx, %rdi
 38787e1:      	callq	0x384cf60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.5209651971967471703>
 38787e6:      	movq	%r14, %rdi
 38787e9:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 38787ee:      	callq	*0x18c85c4(%rip)        # 0x5140db8 <writev+0x5140db8>
 38787f4:      	int3
 38787f5:      	int3
 38787f6:      	int3
 38787f7:      	int3
 38787f8:      	int3
 38787f9:      	int3
 38787fa:      	int3
 38787fb:      	int3
 38787fc:      	int3
 38787fd:      	int3
 38787fe:      	int3
 38787ff:      	int3

0000000003878800 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>:
 3878800:      	pushq	%rbp
 3878801:      	pushq	%r15
 3878803:      	pushq	%r14
 3878805:      	pushq	%rbx
 3878806:      	subq	$0x18, %rsp
 387880a:      	movq	%rdi, %r14
 387880d:      	lock
 387880e:      	incq	0x48(%rdi)
 3878812:      	leaq	0x60(%rdi), %rbx
 3878816:      	movl	$0x1, %ecx
 387881b:      	xorl	%eax, %eax
 387881d:      	lock
 387881e:      	cmpxchgl	%ecx, 0x60(%rdi)
 3878822:      	jne	0x38788e9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xe9>
 3878828:      	movq	0x18c8839(%rip), %r15   # 0x5141068 <writev+0x5141068>
 387882f:      	movq	(%r15), %rax
 3878832:      	shlq	%rax
 3878835:      	testq	%rax, %rax
 3878838:      	jne	0x38788f7 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xf7>
 387883e:      	xorl	%ebp, %ebp
 3878840:      	movzbl	0x64(%r14), %eax
 3878845:      	lock
 3878846:      	incq	0x58(%r14)
 387884a:      	movzbl	0x88(%r14), %eax
 3878852:      	cmpb	$0x3, %al
 3878854:      	je	0x38788be <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xbe>
 3878856:      	cmpb	$0x0, %fs:-0x1c90
 387885f:      	je	0x387886c <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x6c>
 3878861:      	movq	%fs:-0x1c88, %rax
 387886a:      	jmp	0x38788b2 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xb2>
 387886c:      	movl	0x18d567e(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x10>
 3878872:      	testl	%eax, %eax
 3878874:      	jne	0x3878924 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x124>
 387887a:      	movq	0x18d565f(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 3878881:      	movl	0x18d5661(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x8>
 3878887:      	movq	%rax, 0x8(%rsp)
 387888c:      	movl	%ecx, 0x10(%rsp)
 3878890:      	leaq	0x8(%rsp), %rdi
 3878895:      	callq	*0x18c861d(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 387889b:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 38788a2:      	movl	%edx, %eax
 38788a4:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 38788ab:      	shrq	$0x32, %rax
 38788af:      	addq	%rcx, %rax
 38788b2:      	movq	%rax, 0x28(%r14)
 38788b6:      	movb	$0x2, 0x88(%r14)
 38788be:      	testb	%bpl, %bpl
 38788c1:      	jne	0x38788d5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xd5>
 38788c3:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 38788cd:      	movq	(%r15), %rcx
 38788d0:      	testq	%rax, %rcx
 38788d3:      	jne	0x3878913 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x113>
 38788d5:      	xorl	%eax, %eax
 38788d7:      	xchgl	%eax, (%rbx)
 38788d9:      	cmpl	$0x2, %eax
 38788dc:      	je	0x3878908 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x108>
 38788de:      	addq	$0x18, %rsp
 38788e2:      	popq	%rbx
 38788e3:      	popq	%r14
 38788e5:      	popq	%r15
 38788e7:      	popq	%rbp
 38788e8:      	retq
 38788e9:      	movq	%rbx, %rdi
 38788ec:      	callq	*0x18c8b16(%rip)        # 0x5141408 <writev+0x5141408>
 38788f2:      	jmp	0x3878828 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x28>
 38788f7:      	callq	*0x18c877b(%rip)        # 0x5141078 <writev+0x5141078>
 38788fd:      	movl	%eax, %ebp
 38788ff:      	xorb	$0x1, %bpl
 3878903:      	jmp	0x3878840 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x40>
 3878908:      	movq	%rbx, %rdi
 387890b:      	callq	*0x18c875f(%rip)        # 0x5141070 <writev+0x5141070>
 3878911:      	jmp	0x38788de <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xde>
 3878913:      	callq	*0x18c875f(%rip)        # 0x5141078 <writev+0x5141078>
 3878919:      	testb	%al, %al
 387891b:      	jne	0x38788d5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xd5>
 387891d:      	movb	$0x1, 0x64(%r14)
 3878922:      	jmp	0x38788d5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xd5>
 3878924:      	leaq	0x18d55b5(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 387892b:      	callq	0x3b2b87b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 3878930:      	jmp	0x387887a <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x7a>
 3878935:      	movq	%rax, %r14
 3878938:      	movzbl	%bpl, %esi
 387893c:      	movq	%rbx, %rdi
 387893f:      	callq	0x384cf60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.5209651971967471703>
 3878944:      	movq	%r14, %rdi
 3878947:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 387894c:      	callq	*0x18c8466(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878952:      	int3
 3878953:      	int3
 3878954:      	int3
 3878955:      	int3
 3878956:      	int3
 3878957:      	int3
 3878958:      	int3
 3878959:      	int3
 387895a:      	int3
 387895b:      	int3
 387895c:      	int3
 387895d:      	int3
 387895e:      	int3
 387895f:      	int3

0000000003878960 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free>:
 3878960:      	pushq	%rbp
 3878961:      	pushq	%r15
 3878963:      	pushq	%r14
 3878965:      	pushq	%rbx
 3878966:      	subq	$0x18, %rsp
 387896a:      	movq	%rdi, %r14
 387896d:      	leaq	0x60(%rdi), %rbx
 3878971:      	movl	$0x1, %ecx
 3878976:      	xorl	%eax, %eax
 3878978:      	lock
 3878979:      	cmpxchgl	%ecx, 0x60(%rdi)
 387897d:      	jne	0x3878a48 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xe8>
 3878983:      	movq	0x18c86de(%rip), %r15   # 0x5141068 <writev+0x5141068>
 387898a:      	movq	(%r15), %rax
 387898d:      	shlq	%rax
 3878990:      	testq	%rax, %rax
 3878993:      	jne	0x3878a56 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xf6>
 3878999:      	xorl	%ebp, %ebp
 387899b:      	movzbl	0x64(%r14), %eax
 38789a0:      	movq	0x58(%r14), %rax
 38789a4:      	testq	%rax, %rax
 38789a7:      	jne	0x3878a1d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xbd>
 38789a9:      	movzbl	0x88(%r14), %eax
 38789b1:      	cmpb	$0x3, %al
 38789b3:      	je	0x3878a1d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xbd>
 38789b5:      	cmpb	$0x0, %fs:-0x1c90
 38789be:      	je	0x38789cb <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x6b>
 38789c0:      	movq	%fs:-0x1c88, %rax
 38789c9:      	jmp	0x3878a11 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xb1>
 38789cb:      	movl	0x18d551f(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x10>
 38789d1:      	testl	%eax, %eax
 38789d3:      	jne	0x3878a91 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x131>
 38789d9:      	movq	0x18d5500(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 38789e0:      	movl	0x18d5502(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x8>
 38789e6:      	movq	%rax, 0x8(%rsp)
 38789eb:      	movl	%ecx, 0x10(%rsp)
 38789ef:      	leaq	0x8(%rsp), %rdi
 38789f4:      	callq	*0x18c84be(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 38789fa:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 3878a01:      	movl	%edx, %eax
 3878a03:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 3878a0a:      	shrq	$0x32, %rax
 3878a0e:      	addq	%rcx, %rax
 3878a11:      	movq	%rax, 0x28(%r14)
 3878a15:      	movb	$0x1, 0x88(%r14)
 3878a1d:      	testb	%bpl, %bpl
 3878a20:      	jne	0x3878a34 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xd4>
 3878a22:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3878a2c:      	movq	(%r15), %rcx
 3878a2f:      	testq	%rax, %rcx
 3878a32:      	jne	0x3878a80 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x120>
 3878a34:      	xorl	%eax, %eax
 3878a36:      	xchgl	%eax, (%rbx)
 3878a38:      	cmpl	$0x2, %eax
 3878a3b:      	je	0x3878a75 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x115>
 3878a3d:      	addq	$0x18, %rsp
 3878a41:      	popq	%rbx
 3878a42:      	popq	%r14
 3878a44:      	popq	%r15
 3878a46:      	popq	%rbp
 3878a47:      	retq
 3878a48:      	movq	%rbx, %rdi
 3878a4b:      	callq	*0x18c89b7(%rip)        # 0x5141408 <writev+0x5141408>
 3878a51:      	jmp	0x3878983 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x23>
 3878a56:      	callq	*0x18c861c(%rip)        # 0x5141078 <writev+0x5141078>
 3878a5c:      	movl	%eax, %ebp
 3878a5e:      	xorb	$0x1, %bpl
 3878a62:      	movzbl	0x64(%r14), %eax
 3878a67:      	movq	0x58(%r14), %rax
 3878a6b:      	testq	%rax, %rax
 3878a6e:      	jne	0x3878a1d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xbd>
 3878a70:      	jmp	0x38789a9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x49>
 3878a75:      	movq	%rbx, %rdi
 3878a78:      	callq	*0x18c85f2(%rip)        # 0x5141070 <writev+0x5141070>
 3878a7e:      	jmp	0x3878a3d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xdd>
 3878a80:      	callq	*0x18c85f2(%rip)        # 0x5141078 <writev+0x5141078>
 3878a86:      	testb	%al, %al
 3878a88:      	jne	0x3878a34 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xd4>
 3878a8a:      	movb	$0x1, 0x64(%r14)
 3878a8f:      	jmp	0x3878a34 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xd4>
 3878a91:      	leaq	0x18d5448(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 3878a98:      	callq	0x3b2b87b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 3878a9d:      	jmp	0x38789d9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x79>
 3878aa2:      	movq	%rax, %r14
 3878aa5:      	movzbl	%bpl, %esi
 3878aa9:      	movq	%rbx, %rdi
 3878aac:      	callq	0x384cf60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.5209651971967471703>
 3878ab1:      	movq	%r14, %rdi
 3878ab4:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3878ab9:      	callq	*0x18c82f9(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878abf:      	int3

0000000003878c30 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703>:
 3878c30:      	pushq	%rbp
 3878c31:      	pushq	%r15
 3878c33:      	pushq	%r14
 3878c35:      	pushq	%r13
 3878c37:      	pushq	%r12
 3878c39:      	pushq	%rbx
 3878c3a:      	subq	$0xb8, %rsp
 3878c41:      	movq	%rsi, %r14
 3878c44:      	movq	%rdi, %rbx
 3878c47:      	movzbl	0x88(%rsi), %eax
 3878c4e:      	movb	%al, 0x7(%rsp)
 3878c52:      	cmpb	$0x0, %fs:-0x1c90
 3878c5b:      	je	0x3878c68 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x38>
 3878c5d:      	movq	%fs:-0x1c88, %r13
 3878c66:      	jmp	0x3878cae <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x7e>
 3878c68:      	movl	0x18d5282(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x10>
 3878c6e:      	testl	%eax, %eax
 3878c70:      	jne	0x3878f18 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x2e8>
 3878c76:      	movq	0x18d5263(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 3878c7d:      	movl	0x18d5265(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x8>
 3878c83:      	movq	%rax, 0x70(%rsp)
 3878c88:      	movl	%ecx, 0x78(%rsp)
 3878c8c:      	leaq	0x70(%rsp), %rdi
 3878c91:      	callq	*0x18c8221(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 3878c97:      	imulq	$0x3e8, %rax, %rax      # imm = 0x3E8
 3878c9e:      	movl	%edx, %ecx
 3878ca0:      	imulq	$0x431bde83, %rcx, %r13 # imm = 0x431BDE83
 3878ca7:      	shrq	$0x32, %r13
 3878cab:      	addq	%rax, %r13
 3878cae:      	leaq	0x88(%rsp), %rdi
 3878cb6:      	movq	%r14, %rsi
 3878cb9:      	callq	*0x18c82d9(%rip)        # 0x5140f98 <writev+0x5140f98>
 3878cbf:      	movq	0x20(%r14), %rax
 3878cc3:      	movq	%rax, 0x68(%rsp)
 3878cc8:      	movq	0x28(%r14), %rax
 3878ccc:      	movq	%rax, 0x60(%rsp)
 3878cd1:      	movq	0x30(%r14), %rax
 3878cd5:      	testq	%rax, %rax
 3878cd8:      	je	0x3878cfa <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0xca>
 3878cda:      	decq	%rax
 3878cdd:      	xorl	%edx, %edx
 3878cdf:      	movq	%r13, %rcx
 3878ce2:      	subq	%rax, %rcx
 3878ce5:      	cmovaeq	%rcx, %rdx
 3878ce9:      	movq	%rdx, 0x20(%rsp)
 3878cee:      	movl	$0x1, %eax
 3878cf3:      	movq	%rax, 0x18(%rsp)
 3878cf8:      	jmp	0x3878d03 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0xd3>
 3878cfa:      	movq	$0x0, 0x18(%rsp)
 3878d03:      	movq	0x28(%r14), %rax
 3878d07:      	movq	%rax, 0x38(%rsp)
 3878d0c:      	movq	0x30(%r14), %rbp
 3878d10:      	testq	%rbp, %rbp
 3878d13:      	je	0x3878d28 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0xf8>
 3878d15:      	leaq	-0x1(%rbp), %rax
 3878d19:      	xorl	%r15d, %r15d
 3878d1c:      	movq	%r13, %rcx
 3878d1f:      	subq	%rax, %rcx
 3878d22:      	cmovaeq	%rcx, %r15
 3878d26:      	jmp	0x3878d28 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0xf8>
 3878d28:      	movq	0x38(%r14), %rax
 3878d2c:      	movq	%rax, 0x58(%rsp)
 3878d31:      	movq	0x40(%r14), %rax
 3878d35:      	movq	%rax, 0x50(%rsp)
 3878d3a:      	movq	0x48(%r14), %rax
 3878d3e:      	movq	%rax, 0x48(%rsp)
 3878d43:      	movq	0x50(%r14), %rax
 3878d47:      	movq	%rax, 0x40(%rsp)
 3878d4c:      	movq	0x18(%r14), %rax
 3878d50:      	movq	0x30(%rax), %rcx
 3878d54:      	movq	%rcx, 0x30(%rsp)
 3878d59:      	movq	0x38(%rax), %r12
 3878d5d:      	movq	0x40(%rax), %rax
 3878d61:      	movq	%rax, 0x28(%rsp)
 3878d66:      	movq	$-0x1, 0x8(%rsp)
 3878d6f:      	cmpb	$0x3, 0x7(%rsp)
 3878d74:      	jne	0x3878e4e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x21e>
 3878d7a:      	movl	$0x1, %ecx
 3878d7f:      	xorl	%eax, %eax
 3878d81:      	lock
 3878d82:      	cmpxchgl	%ecx, 0x68(%r14)
 3878d87:      	jne	0x3878e4e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x21e>
 3878d8d:      	movq	0x18c82d4(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3878d94:      	movq	(%rax), %rax
 3878d97:      	shlq	%rax
 3878d9a:      	testq	%rax, %rax
 3878d9d:      	jne	0x3878f29 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x2f9>
 3878da3:      	xorl	%ecx, %ecx
 3878da5:      	leaq	0x68(%r14), %rax
 3878da9:      	movq	%rax, 0x10(%rsp)
 3878dae:      	movabsq	$0x7fffffffffffffff, %rdx # imm = 0x7FFFFFFFFFFFFFFF
 3878db8:      	movzbl	0x6c(%r14), %eax
 3878dbd:      	testb	%al, %al
 3878dbf:      	je	0x3878dee <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1be>
 3878dc1:      	testb	%cl, %cl
 3878dc3:      	jne	0x3878dd8 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1a8>
 3878dc5:      	movq	0x18c829c(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3878dcc:      	movq	(%rax), %rax
 3878dcf:      	testq	%rdx, %rax
 3878dd2:      	jne	0x3878f44 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x314>
 3878dd8:      	xorl	%eax, %eax
 3878dda:      	movq	0x10(%rsp), %rdi
 3878ddf:      	xchgl	%eax, (%rdi)
 3878de1:      	cmpl	$0x2, %eax
 3878de4:      	jne	0x3878e4e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x21e>
 3878de6:      	callq	*0x18c8284(%rip)        # 0x5141070 <writev+0x5141070>
 3878dec:      	jmp	0x3878e4e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x21e>
 3878dee:      	movl	%ecx, 0x8(%rsp)
 3878df2:      	leaq	0x70(%r14), %rsi
 3878df6:      	leaq	0x70(%rsp), %rdi
 3878dfb:      	callq	*0x18c8197(%rip)        # 0x5140f98 <writev+0x5140f98>
 3878e01:      	cmpb	$0x0, 0x8(%rsp)
 3878e06:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3878e10:      	jne	0x3878e25 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1f5>
 3878e12:      	movq	0x18c824f(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3878e19:      	movq	(%rax), %rax
 3878e1c:      	testq	%rcx, %rax
 3878e1f:      	jne	0x3878f5c <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x32c>
 3878e25:      	xorl	%eax, %eax
 3878e27:      	movq	0x10(%rsp), %rdi
 3878e2c:      	xchgl	%eax, (%rdi)
 3878e2e:      	cmpl	$0x2, %eax
 3878e31:      	je	0x3878f39 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x309>
 3878e37:      	movq	0x70(%rsp), %rax
 3878e3c:      	movq	%rax, 0x8(%rsp)
 3878e41:      	movups	0x78(%rsp), %xmm0
 3878e46:      	movaps	%xmm0, 0xa0(%rsp)
 3878e4e:      	addq	0x30(%rsp), %r12
 3878e53:      	addq	0x28(%rsp), %r12
 3878e58:      	xorl	%eax, %eax
 3878e5a:      	movq	%r13, %rcx
 3878e5d:      	subq	0x38(%rsp), %rcx
 3878e62:      	cmovbq	%rax, %rcx
 3878e66:      	cmpq	%rcx, %r15
 3878e69:      	cmovaeq	%rcx, %r15
 3878e6d:      	testq	%rbp, %rbp
 3878e70:      	cmoveq	%rcx, %r15
 3878e74:      	movq	%r13, %rcx
 3878e77:      	subq	0x60(%rsp), %rcx
 3878e7c:      	cmovbq	%rax, %rcx
 3878e80:      	subq	0x68(%rsp), %r13
 3878e85:      	cmovbq	%rax, %r13
 3878e89:      	movq	0x98(%rsp), %rax
 3878e91:      	movq	%rax, 0x20(%rbx)
 3878e95:      	movups	0x88(%rsp), %xmm0
 3878e9d:      	movups	%xmm0, 0x10(%rbx)
 3878ea1:      	movzbl	0x7(%rsp), %eax
 3878ea6:      	movb	%al, 0x80(%rbx)
 3878eac:      	movq	%r13, 0x40(%rbx)
 3878eb0:      	movq	%rcx, 0x48(%rbx)
 3878eb4:      	movq	0x18(%rsp), %rax
 3878eb9:      	movq	%rax, (%rbx)
 3878ebc:      	movq	0x20(%rsp), %rax
 3878ec1:      	movq	%rax, 0x8(%rbx)
 3878ec5:      	movq	%r15, 0x50(%rbx)
 3878ec9:      	movq	0x58(%rsp), %rax
 3878ece:      	movq	%rax, 0x58(%rbx)
 3878ed2:      	movq	0x50(%rsp), %rax
 3878ed7:      	movq	%rax, 0x60(%rbx)
 3878edb:      	movq	0x48(%rsp), %rax
 3878ee0:      	movq	%rax, 0x68(%rbx)
 3878ee4:      	movq	0x40(%rsp), %rax
 3878ee9:      	movq	%rax, 0x70(%rbx)
 3878eed:      	movq	%r12, 0x78(%rbx)
 3878ef1:      	movq	0x8(%rsp), %rax
 3878ef6:      	movq	%rax, 0x28(%rbx)
 3878efa:      	movaps	0xa0(%rsp), %xmm0
 3878f02:      	movups	%xmm0, 0x30(%rbx)
 3878f06:      	addq	$0xb8, %rsp
 3878f0d:      	popq	%rbx
 3878f0e:      	popq	%r12
 3878f10:      	popq	%r13
 3878f12:      	popq	%r14
 3878f14:      	popq	%r15
 3878f16:      	popq	%rbp
 3878f17:      	retq
 3878f18:      	leaq	0x18d4fc1(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 3878f1f:      	callq	0x3b2b87b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 3878f24:      	jmp	0x3878c76 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x46>
 3878f29:      	callq	*0x18c8149(%rip)        # 0x5141078 <writev+0x5141078>
 3878f2f:      	movl	%eax, %ecx
 3878f31:      	xorb	$0x1, %cl
 3878f34:      	jmp	0x3878da5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x175>
 3878f39:      	callq	*0x18c8131(%rip)        # 0x5141070 <writev+0x5141070>
 3878f3f:      	jmp	0x3878e37 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x207>
 3878f44:      	callq	*0x18c812e(%rip)        # 0x5141078 <writev+0x5141078>
 3878f4a:      	testb	%al, %al
 3878f4c:      	jne	0x3878dd8 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1a8>
 3878f52:      	movb	$0x1, 0x6c(%r14)
 3878f57:      	jmp	0x3878dd8 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1a8>
 3878f5c:      	callq	*0x18c8116(%rip)        # 0x5141078 <writev+0x5141078>
 3878f62:      	testb	%al, %al
 3878f64:      	jne	0x3878e25 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1f5>
 3878f6a:      	movb	$0x1, 0x6c(%r14)
 3878f6f:      	jmp	0x3878e25 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x1f5>
 3878f74:      	movq	%rax, %rbx
 3878f77:      	jmp	0x3878f8b <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x35b>
 3878f79:      	movq	%rax, %rbx
 3878f7c:      	movzbl	0x8(%rsp), %esi
 3878f81:      	movq	0x10(%rsp), %rdi
 3878f86:      	callq	0x384cf10 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCscdodAO9FK5_5alloc6string6StringEECs3pwlnhBXFtN_12memra_server>
 3878f8b:      	movq	0x88(%rsp), %rsi
 3878f93:      	testq	%rsi, %rsi
 3878f96:      	je	0x3878fab <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703+0x37b>
 3878f98:      	movq	0x90(%rsp), %rdi
 3878fa0:      	movl	$0x1, %edx
 3878fa5:      	callq	*0x18c7d95(%rip)        # 0x5140d40 <writev+0x5140d40>
 3878fab:      	movq	%rbx, %rdi
 3878fae:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3878fb3:      	callq	*0x18c7dff(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878fb9:      	int3
 3878fba:      	int3
 3878fbb:      	int3
 3878fbc:      	int3
 3878fbd:      	int3
 3878fbe:      	int3
 3878fbf:      	int3

000000000388bef0 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends>:
 388bef0:      	pushq	%rbx
 388bef1:      	subq	$0xa0, %rsp
 388bef8:      	movl	$0xea60, %edi           # imm = 0xEA60
 388befd:      	callq	0x3875400 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms>
 388bf02:      	movq	%rax, %rbx
 388bf05:      	movq	%rax, 0x10(%rsp)
 388bf0a:      	leaq	-0x3500fb1(%rip), %rdi  # 0x38af60 <anon.985887fba90c90e54334d6f2bee737a0.4512.llvm.9834526418542584810+0x20>
 388bf11:      	movl	$0x8, %esi
 388bf16:      	movl	$0x1, %edx
 388bf1b:      	callq	0x3980bb0 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_>
 388bf20:      	addq	$0x10, %rbx
 388bf24:      	leaq	-0x3500fcb(%rip), %rsi  # 0x38af60 <anon.985887fba90c90e54334d6f2bee737a0.4512.llvm.9834526418542584810+0x20>
 388bf2b:      	movl	$0x8, %edx
 388bf30:      	movq	%rbx, %rdi
 388bf33:      	movq	%rax, %rcx
 388bf36:      	callq	0x3875610 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route>
 388bf3b:      	movq	%rax, %rbx
 388bf3e:      	movq	%rax, 0x8(%rsp)
 388bf43:      	movzbl	0x98(%rax), %eax
 388bf4a:      	cmpb	$0x3, %al
 388bf4c:      	je	0x388bfb5 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0xc5>
 388bf4e:      	cmpb	$0x0, %fs:-0x1c90
 388bf57:      	je	0x388bf64 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x74>
 388bf59:      	movq	%fs:-0x1c88, %rax
 388bf62:      	jmp	0x388bfaa <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0xba>
 388bf64:      	movl	0x18c1f86(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x10>
 388bf6a:      	testl	%eax, %eax
 388bf6c:      	jne	0x388c25f <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36f>
 388bf72:      	movq	0x18c1f67(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 388bf79:      	movl	0x18c1f69(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703+0x8>
 388bf7f:      	movq	%rax, 0x18(%rsp)
 388bf84:      	movl	%ecx, 0x20(%rsp)
 388bf88:      	leaq	0x18(%rsp), %rdi
 388bf8d:      	callq	*0x18b4f25(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 388bf93:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 388bf9a:      	movl	%edx, %eax
 388bf9c:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 388bfa3:      	shrq	$0x32, %rax
 388bfa7:      	addq	%rcx, %rax
 388bfaa:      	movq	%rax, 0x38(%rbx)
 388bfae:      	movb	$0x1, 0x98(%rbx)
 388bfb5:      	movq	0x8(%rsp), %rdi
 388bfba:      	addq	$0x10, %rdi
 388bfbe:      	callq	0x3878800 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>
 388bfc3:      	movq	0x8(%rsp), %rdi
 388bfc8:      	addq	$0x10, %rdi
 388bfcc:      	callq	0x3878800 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>
 388bfd1:      	movq	0x8(%rsp), %rdi
 388bfd6:      	addq	$0x10, %rdi
 388bfda:      	callq	0x3878690 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>
 388bfdf:      	movq	0x8(%rsp), %rdi
 388bfe4:      	addq	$0x10, %rdi
 388bfe8:      	callq	0x3878960 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free>
 388bfed:      	movq	0x8(%rsp), %rsi
 388bff2:      	addq	$0x10, %rsi
 388bff6:      	leaq	0x18(%rsp), %rdi
 388bffb:      	callq	0x3878c30 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703>
 388c000:      	cmpb	$0x2, 0x98(%rsp)
 388c008:      	jne	0x388c1b6 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2c6>
 388c00e:      	movq	0x28(%rsp), %rsi
 388c013:      	testq	%rsi, %rsi
 388c016:      	je	0x388c028 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x138>
 388c018:      	movq	0x30(%rsp), %rdi
 388c01d:      	movl	$0x1, %edx
 388c022:      	callq	*0x18b4d18(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c028:      	movq	0x40(%rsp), %rsi
 388c02d:      	cmpq	$-0x1, %rsi
 388c031:      	je	0x388c048 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x158>
 388c033:      	testq	%rsi, %rsi
 388c036:      	je	0x388c048 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x158>
 388c038:      	movq	0x48(%rsp), %rdi
 388c03d:      	movl	$0x1, %edx
 388c042:      	callq	*0x18b4cf8(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c048:      	movq	0x8(%rsp), %rdi
 388c04d:      	addq	$0x10, %rdi
 388c051:      	callq	0x3878690 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>
 388c056:      	movq	0x8(%rsp), %rsi
 388c05b:      	addq	$0x10, %rsi
 388c05f:      	leaq	0x18(%rsp), %rdi
 388c064:      	callq	0x3878c30 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703>
 388c069:      	cmpb	$0x1, 0x98(%rsp)
 388c071:      	jne	0x388c1e3 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2f3>
 388c077:      	movq	0x28(%rsp), %rsi
 388c07c:      	testq	%rsi, %rsi
 388c07f:      	je	0x388c091 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x1a1>
 388c081:      	movq	0x30(%rsp), %rdi
 388c086:      	movl	$0x1, %edx
 388c08b:      	callq	*0x18b4caf(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c091:      	movq	0x40(%rsp), %rsi
 388c096:      	cmpq	$-0x1, %rsi
 388c09a:      	je	0x388c0b1 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x1c1>
 388c09c:      	testq	%rsi, %rsi
 388c09f:      	je	0x388c0b1 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x1c1>
 388c0a1:      	movq	0x48(%rsp), %rdi
 388c0a6:      	movl	$0x1, %edx
 388c0ab:      	callq	*0x18b4c8f(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c0b1:      	movq	0x8(%rsp), %rdi
 388c0b6:      	addq	$0x10, %rdi
 388c0ba:      	callq	0x3878690 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>
 388c0bf:      	movq	0x8(%rsp), %rsi
 388c0c4:      	addq	$0x10, %rsi
 388c0c8:      	leaq	0x18(%rsp), %rdi
 388c0cd:      	callq	0x3878c30 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703>
 388c0d2:      	cmpb	$0x1, 0x98(%rsp)
 388c0da:      	jne	0x388c205 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x315>
 388c0e0:      	movq	0x28(%rsp), %rsi
 388c0e5:      	testq	%rsi, %rsi
 388c0e8:      	je	0x388c0fa <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x20a>
 388c0ea:      	movq	0x30(%rsp), %rdi
 388c0ef:      	movl	$0x1, %edx
 388c0f4:      	callq	*0x18b4c46(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c0fa:      	movq	0x40(%rsp), %rsi
 388c0ff:      	cmpq	$-0x1, %rsi
 388c103:      	je	0x388c11a <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x22a>
 388c105:      	testq	%rsi, %rsi
 388c108:      	je	0x388c11a <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x22a>
 388c10a:      	movq	0x48(%rsp), %rdi
 388c10f:      	movl	$0x1, %edx
 388c114:      	callq	*0x18b4c26(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c11a:      	movq	0x8(%rsp), %rdi
 388c11f:      	addq	$0x10, %rdi
 388c123:      	callq	0x3878800 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>
 388c128:      	movq	0x8(%rsp), %rsi
 388c12d:      	addq	$0x10, %rsi
 388c131:      	leaq	0x18(%rsp), %rdi
 388c136:      	callq	0x3878c30 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.5209651971967471703>
 388c13b:      	cmpb	$0x2, 0x98(%rsp)
 388c143:      	jne	0x388c232 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x342>
 388c149:      	movq	0x28(%rsp), %rsi
 388c14e:      	testq	%rsi, %rsi
 388c151:      	je	0x388c163 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x273>
 388c153:      	movq	0x30(%rsp), %rdi
 388c158:      	movl	$0x1, %edx
 388c15d:      	callq	*0x18b4bdd(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c163:      	movq	0x40(%rsp), %rsi
 388c168:      	cmpq	$-0x1, %rsi
 388c16c:      	je	0x388c183 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x293>
 388c16e:      	testq	%rsi, %rsi
 388c171:      	je	0x388c183 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x293>
 388c173:      	movq	0x48(%rsp), %rdi
 388c178:      	movl	$0x1, %edx
 388c17d:      	callq	*0x18b4bbd(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c183:      	movq	0x8(%rsp), %rax
 388c188:      	lock
 388c189:      	decq	(%rax)
 388c18c:      	jne	0x388c198 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2a8>
 388c18e:      	leaq	0x8(%rsp), %rdi
 388c193:      	callq	0x39bcef0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 388c198:      	movq	0x10(%rsp), %rax
 388c19d:      	lock
 388c19e:      	decq	(%rax)
 388c1a1:      	jne	0x388c1ad <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2bd>
 388c1a3:      	leaq	0x10(%rsp), %rdi
 388c1a8:      	callq	0x39bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 388c1ad:      	addq	$0xa0, %rsp
 388c1b4:      	popq	%rbx
 388c1b5:      	retq
 388c1b6:      	leaq	0x98(%rsp), %rsi
 388c1be:      	leaq	-0x2ec871a(%rip), %rdx  # 0x9c3aab <anon.bd10a02e3ec00da32c2ae70ffdaca96c.608.llvm.5209651971967471703+0x1888e>
 388c1c5:      	leaq	-0x2ec7b20(%rip), %rcx  # 0x9c46ac <anon.bd10a02e3ec00da32c2ae70ffdaca96c.779.llvm.5209651971967471703+0x430>
 388c1cc:      	leaq	0x178ad8d(%rip), %r9    # 0x5016f60 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.5209651971967471703+0x3f0>
 388c1d3:      	movl	$0x3d, %r8d
 388c1d9:      	xorl	%edi, %edi
 388c1db:      	callq	*0x18b6267(%rip)        # 0x5142448 <writev+0x5142448>
 388c1e1:      	jmp	0x388c25d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36d>
 388c1e3:      	leaq	0x98(%rsp), %rsi
 388c1eb:      	leaq	-0x2ee23f8(%rip), %rdx  # 0x9a9dfa <anon.985887fba90c90e54334d6f2bee737a0.8597.llvm.9834526418542584810+0xf40>
 388c1f2:      	leaq	0x178ad4f(%rip), %r9    # 0x5016f48 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.5209651971967471703+0x3d8>
 388c1f9:      	xorl	%edi, %edi
 388c1fb:      	xorl	%ecx, %ecx
 388c1fd:      	callq	*0x18b6245(%rip)        # 0x5142448 <writev+0x5142448>
 388c203:      	jmp	0x388c25d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36d>
 388c205:      	leaq	0x98(%rsp), %rsi
 388c20d:      	leaq	-0x2ee241a(%rip), %rdx  # 0x9a9dfa <anon.985887fba90c90e54334d6f2bee737a0.8597.llvm.9834526418542584810+0xf40>
 388c214:      	leaq	-0x2ec7b8d(%rip), %rcx  # 0x9c468e <anon.bd10a02e3ec00da32c2ae70ffdaca96c.779.llvm.5209651971967471703+0x412>
 388c21b:      	leaq	0x178ad0e(%rip), %r9    # 0x5016f30 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.5209651971967471703+0x3c0>
 388c222:      	movl	$0x3d, %r8d
 388c228:      	xorl	%edi, %edi
 388c22a:      	callq	*0x18b6218(%rip)        # 0x5142448 <writev+0x5142448>
 388c230:      	jmp	0x388c25d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36d>
 388c232:      	leaq	0x98(%rsp), %rsi
 388c23a:      	leaq	-0x2ec8796(%rip), %rdx  # 0x9c3aab <anon.bd10a02e3ec00da32c2ae70ffdaca96c.608.llvm.5209651971967471703+0x1888e>
 388c241:      	leaq	-0x2ec7bd7(%rip), %rcx  # 0x9c4671 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.779.llvm.5209651971967471703+0x3f5>
 388c248:      	leaq	0x178acc9(%rip), %r9    # 0x5016f18 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.5209651971967471703+0x3a8>
 388c24f:      	movl	$0x3b, %r8d
 388c255:      	xorl	%edi, %edi
 388c257:      	callq	*0x18b61eb(%rip)        # 0x5142448 <writev+0x5142448>
 388c25d:      	ud2
 388c25f:      	leaq	0x18c1c7a(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.5209651971967471703>
 388c266:      	callq	0x3b2b87b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 388c26b:      	jmp	0x388bf72 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x82>
 388c270:      	movq	%rax, %rbx
 388c273:      	jmp	0x388c2a2 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x3b2>
 388c275:      	movq	%rax, %rbx
 388c278:      	jmp	0x388c28d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x39d>
 388c27a:      	jmp	0x388c280 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x390>
 388c27c:      	jmp	0x388c280 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x390>
 388c27e:      	jmp	0x388c280 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x390>
 388c280:      	movq	%rax, %rbx
 388c283:      	leaq	0x18(%rsp), %rdi
 388c288:      	callq	0x384d580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_>
 388c28d:      	movq	0x8(%rsp), %rax
 388c292:      	lock
 388c293:      	decq	(%rax)
 388c296:      	jne	0x388c2a2 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x3b2>
 388c298:      	leaq	0x8(%rsp), %rdi
 388c29d:      	callq	0x39bcef0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 388c2a2:      	movq	0x10(%rsp), %rax
 388c2a7:      	lock
 388c2a8:      	decq	(%rax)
 388c2ab:      	jne	0x388c2b7 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x3c7>
 388c2ad:      	leaq	0x10(%rsp), %rdi
 388c2b2:      	callq	0x39bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 388c2b7:      	movq	%rbx, %rdi
 388c2ba:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 388c2bf:      	callq	*0x18b4af3(%rip)        # 0x5140db8 <writev+0x5140db8>
 388c2c5:      	int3
 388c2c6:      	int3
 388c2c7:      	int3
 388c2c8:      	int3
 388c2c9:      	int3
 388c2ca:      	int3
 388c2cb:      	int3
 388c2cc:      	int3
 388c2cd:      	int3
 388c2ce:      	int3
 388c2cf:      	int3

0000000003980bb0 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_>:
 3980bb0:      	pushq	%r15
 3980bb2:      	pushq	%r14
 3980bb4:      	pushq	%r12
 3980bb6:      	pushq	%rbx
 3980bb7:      	subq	$0x2c8, %rsp            # imm = 0x2C8
 3980bbe:      	movq	%rdx, %r15
 3980bc1:      	movq	%rsi, %rbx
 3980bc4:      	testq	%rsi, %rsi
 3980bc7:      	je	0x3980bfd <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x4d>
 3980bc9:      	movq	%rdi, %r12
 3980bcc:      	callq	*0x17c01be(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980bd2:      	movl	$0x1, %esi
 3980bd7:      	movq	%rbx, %rdi
 3980bda:      	callq	*0x17c01b8(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980be0:      	testq	%rax, %rax
 3980be3:      	je	0x3980e7b <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2cb>
 3980be9:      	movq	%rax, %r14
 3980bec:      	movq	%rax, %rdi
 3980bef:      	movq	%r12, %rsi
 3980bf2:      	movq	%rbx, %rdx
 3980bf5:      	callq	*0x17c013d(%rip)        # 0x5140d38 <writev+0x5140d38>
 3980bfb:      	jmp	0x3980c03 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x53>
 3980bfd:      	movl	$0x1, %r14d
 3980c03:      	cmpq	$0x1, %r15
 3980c07:      	adcq	$0x0, %r15
 3980c0b:      	callq	*0x17c017f(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980c11:      	movl	$0x200, %edi            # imm = 0x200
 3980c16:      	movl	$0x8, %esi
 3980c1b:      	callq	*0x17c0177(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980c21:      	testq	%rax, %rax
 3980c24:      	je	0x3980e57 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2a7>
 3980c2a:      	movq	%rax, %r12
 3980c2d:      	callq	*0x17c015d(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980c33:      	movl	$0x800, %edi            # imm = 0x800
 3980c38:      	movl	$0x8, %esi
 3980c3d:      	callq	*0x17c0155(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980c43:      	testq	%rax, %rax
 3980c46:      	je	0x3980e69 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2b9>
 3980c4c:      	xorps	%xmm0, %xmm0
 3980c4f:      	movups	%xmm0, 0x100(%rsp)
 3980c57:      	movups	%xmm0, 0x118(%rsp)
 3980c5f:      	movups	%xmm0, 0x128(%rsp)
 3980c67:      	movups	%xmm0, 0x138(%rsp)
 3980c6f:      	movups	%xmm0, 0x148(%rsp)
 3980c77:      	movups	%xmm0, 0x158(%rsp)
 3980c7f:      	movups	%xmm0, 0x168(%rsp)
 3980c87:      	movups	%xmm0, 0x178(%rsp)
 3980c8f:      	movups	%xmm0, 0x188(%rsp)
 3980c97:      	movups	%xmm0, 0x198(%rsp)
 3980c9f:      	movups	%xmm0, 0x1a8(%rsp)
 3980ca7:      	movups	%xmm0, 0x1b8(%rsp)
 3980caf:      	movups	%xmm0, 0x1c8(%rsp)
 3980cb7:      	movups	%xmm0, 0x1d8(%rsp)
 3980cbf:      	movups	%xmm0, 0x1e8(%rsp)
 3980cc7:      	movups	%xmm0, 0x1f8(%rsp)
 3980ccf:      	movups	%xmm0, 0x208(%rsp)
 3980cd7:      	movups	%xmm0, 0x218(%rsp)
 3980cdf:      	movups	%xmm0, 0x228(%rsp)
 3980ce7:      	movups	%xmm0, 0x2b8(%rsp)
 3980cef:      	movups	%xmm0, 0x2a8(%rsp)
 3980cf7:      	movups	%xmm0, 0x298(%rsp)
 3980cff:      	movups	%xmm0, 0x288(%rsp)
 3980d07:      	movups	%xmm0, 0x278(%rsp)
 3980d0f:      	movups	%xmm0, 0x268(%rsp)
 3980d17:      	movups	%xmm0, 0x258(%rsp)
 3980d1f:      	movups	%xmm0, 0x248(%rsp)
 3980d27:      	movups	%xmm0, 0x238(%rsp)
 3980d2f:      	movq	%rbx, 0x10(%rsp)
 3980d34:      	movq	%r14, 0x18(%rsp)
 3980d39:      	movq	%rbx, 0x20(%rsp)
 3980d3e:      	movl	$0x0, 0xb8(%rsp)
 3980d49:      	movb	$0x0, 0xbc(%rsp)
 3980d51:      	movups	%xmm0, 0x30(%rsp)
 3980d56:      	movups	%xmm0, 0x40(%rsp)
 3980d5b:      	movups	%xmm0, 0x50(%rsp)
 3980d60:      	movq	$0x40, 0xc0(%rsp)
 3980d6c:      	movq	%r12, 0xc8(%rsp)
 3980d74:      	movups	%xmm0, 0xd0(%rsp)
 3980d7c:      	movq	$0x40, 0xe0(%rsp)
 3980d88:      	movl	$0x0, 0xe8(%rsp)
 3980d93:      	movb	$0x0, 0xec(%rsp)
 3980d9b:      	movq	$0x100, 0xf0(%rsp)      # imm = 0x100
 3980da7:      	movq	%rax, 0xf8(%rsp)
 3980daf:      	movq	$0x100, 0x110(%rsp)     # imm = 0x100
 3980dbb:      	movq	$0x1, (%rsp)
 3980dc3:      	movq	$0x1, 0x8(%rsp)
 3980dcc:      	movq	%r15, 0x28(%rsp)
 3980dd1:      	movups	%xmm0, 0x60(%rsp)
 3980dd6:      	movups	%xmm0, 0x70(%rsp)
 3980ddb:      	movups	%xmm0, 0x80(%rsp)
 3980de3:      	movups	%xmm0, 0x90(%rsp)
 3980deb:      	movups	%xmm0, 0xa0(%rsp)
 3980df3:      	movq	$0x0, 0xb0(%rsp)
 3980dff:      	callq	*0x17bff8b(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980e05:      	movl	$0x2c8, %edi            # imm = 0x2C8
 3980e0a:      	movl	$0x8, %esi
 3980e0f:      	callq	*0x17bff83(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980e15:      	testq	%rax, %rax
 3980e18:      	je	0x3980e40 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x290>
 3980e1a:      	movq	%rsp, %rsi
 3980e1d:      	movl	$0x2c8, %edx            # imm = 0x2C8
 3980e22:      	movq	%rax, %rdi
 3980e25:      	movq	%rax, %rbx
 3980e28:      	callq	*0x17bff0a(%rip)        # 0x5140d38 <writev+0x5140d38>
 3980e2e:      	movq	%rbx, %rax
 3980e31:      	addq	$0x2c8, %rsp            # imm = 0x2C8
 3980e38:      	popq	%rbx
 3980e39:      	popq	%r12
 3980e3b:      	popq	%r14
 3980e3d:      	popq	%r15
 3980e3f:      	retq
 3980e40:      	leaq	0x10(%rsp), %rbx
 3980e45:      	movl	$0x8, %edi
 3980e4a:      	movl	$0x2c8, %esi            # imm = 0x2C8
 3980e4f:      	callq	*0x17c010b(%rip)        # 0x5140f60 <writev+0x5140f60>
 3980e55:      	jmp	0x3980e79 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2c9>
 3980e57:      	movl	$0x8, %edi
 3980e5c:      	movl	$0x200, %esi            # imm = 0x200
 3980e61:      	callq	*0x17bff21(%rip)        # 0x5140d88 <writev+0x5140d88>
 3980e67:      	jmp	0x3980e79 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2c9>
 3980e69:      	movl	$0x8, %edi
 3980e6e:      	movl	$0x800, %esi            # imm = 0x800
 3980e73:      	callq	*0x17bff0f(%rip)        # 0x5140d88 <writev+0x5140d88>
 3980e79:      	ud2
 3980e7b:      	movl	$0x1, %edi
 3980e80:      	movq	%rbx, %rsi
 3980e83:      	callq	*0x17bfeff(%rip)        # 0x5140d88 <writev+0x5140d88>
 3980e89:      	movq	%rax, %r15
 3980e8c:      	movl	$0x200, %esi            # imm = 0x200
 3980e91:      	movl	$0x8, %edx
 3980e96:      	movq	%r12, %rdi
 3980e99:      	callq	*0x17bfea1(%rip)        # 0x5140d40 <writev+0x5140d40>
 3980e9f:      	jmp	0x3980ea4 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2f4>
 3980ea1:      	movq	%rax, %r15
 3980ea4:      	testq	%rbx, %rbx
 3980ea7:      	je	0x3980ecd <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x31d>
 3980ea9:      	movl	$0x1, %edx
 3980eae:      	movq	%r14, %rdi
 3980eb1:      	movq	%rbx, %rsi
 3980eb4:      	callq	*0x17bfe86(%rip)        # 0x5140d40 <writev+0x5140d40>
 3980eba:      	movq	%r15, %rdi
 3980ebd:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3980ec2:      	movq	%rax, %r15
 3980ec5:      	movq	%rbx, %rdi
 3980ec8:      	callq	0x398c5d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadEBF_>
 3980ecd:      	movq	%r15, %rdi
 3980ed0:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3980ed5:      	int3
 3980ed6:      	int3
 3980ed7:      	int3
 3980ed8:      	int3
 3980ed9:      	int3
 3980eda:      	int3
 3980edb:      	int3
 3980edc:      	int3
 3980edd:      	int3
 3980ede:      	int3
 3980edf:      	int3

00000000039bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>:
 39bcc60:      	pushq	%rbx
 39bcc61:      	movq	(%rdi), %rbx
 39bcc64:      	movq	0x10(%rbx), %rsi
 39bcc68:      	testq	%rsi, %rsi
 39bcc6b:      	je	0x39bcc7c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x1c>
 39bcc6d:      	movq	0x18(%rbx), %rdi
 39bcc71:      	movl	$0x1, %edx
 39bcc76:      	callq	*0x17840c4(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcc7c:      	movq	0xc0(%rbx), %rsi
 39bcc83:      	testq	%rsi, %rsi
 39bcc86:      	je	0x39bcc9e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x3e>
 39bcc88:      	movq	0xc8(%rbx), %rdi
 39bcc8f:      	shlq	$0x3, %rsi
 39bcc93:      	movl	$0x8, %edx
 39bcc98:      	callq	*0x17840a2(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcc9e:      	movq	0xf0(%rbx), %rsi
 39bcca5:      	testq	%rsi, %rsi
 39bcca8:      	je	0x39bccc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x60>
 39bccaa:      	movq	0xf8(%rbx), %rdi
 39bccb1:      	shlq	$0x3, %rsi
 39bccb5:      	movl	$0x8, %edx
 39bccba:      	callq	*0x1784080(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bccc0:      	cmpq	$-0x1, %rbx
 39bccc4:      	je	0x39bcce1 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x81>
 39bccc6:      	lock
 39bccc7:      	decq	0x8(%rbx)
 39bcccb:      	jne	0x39bcce1 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x81>
 39bcccd:      	movl	$0x2c8, %esi            # imm = 0x2C8
 39bccd2:      	movl	$0x8, %edx
 39bccd7:      	movq	%rbx, %rdi
 39bccda:      	popq	%rbx
 39bccdb:      	jmpq	*0x178405f(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcce1:      	popq	%rbx
 39bcce2:      	retq
 39bcce3:      	int3
 39bcce4:      	int3
 39bcce5:      	int3
 39bcce6:      	int3
 39bcce7:      	int3
 39bcce8:      	int3
 39bcce9:      	int3
 39bccea:      	int3
 39bcceb:      	int3
 39bccec:      	int3
 39bcced:      	int3
 39bccee:      	int3
 39bccef:      	int3

00000000039bcef0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>:
 39bcef0:      	pushq	%rbx
 39bcef1:      	movq	(%rdi), %rbx
 39bcef4:      	movq	0x10(%rbx), %rsi
 39bcef8:      	testq	%rsi, %rsi
 39bcefb:      	je	0x39bcf0c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x1c>
 39bcefd:      	movq	0x18(%rbx), %rdi
 39bcf01:      	movl	$0x1, %edx
 39bcf06:      	callq	*0x1783e34(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf0c:      	movq	0x80(%rbx), %rsi
 39bcf13:      	testq	%rsi, %rsi
 39bcf16:      	je	0x39bcf2a <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x3a>
 39bcf18:      	movq	0x88(%rbx), %rdi
 39bcf1f:      	movl	$0x1, %edx
 39bcf24:      	callq	*0x1783e16(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf2a:      	movq	0x28(%rbx), %rax
 39bcf2e:      	lock
 39bcf2f:      	decq	(%rax)
 39bcf32:      	jne	0x39bcf3d <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x4d>
 39bcf34:      	leaq	0x28(%rbx), %rdi
 39bcf38:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 39bcf3d:      	cmpq	$-0x1, %rbx
 39bcf41:      	je	0x39bcf5e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x6e>
 39bcf43:      	lock
 39bcf44:      	decq	0x8(%rbx)
 39bcf48:      	jne	0x39bcf5e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x6e>
 39bcf4a:      	movl	$0xa0, %esi
 39bcf4f:      	movl	$0x8, %edx
 39bcf54:      	movq	%rbx, %rdi
 39bcf57:      	popq	%rbx
 39bcf58:      	jmpq	*0x1783de2(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf5e:      	popq	%rbx
 39bcf5f:      	retq

00000000039bcf60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>:
 39bcf60:      	pushq	%r15
 39bcf62:      	pushq	%r14
 39bcf64:      	pushq	%rbx
 39bcf65:      	movq	(%rdi), %rbx
 39bcf68:      	movq	0x18(%rbx), %rsi
 39bcf6c:      	testq	%rsi, %rsi
 39bcf6f:      	je	0x39bcf80 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x20>
 39bcf71:      	movq	0x20(%rbx), %rdi
 39bcf75:      	movl	$0x1, %edx
 39bcf7a:      	callq	*0x1783dc0(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf80:      	movq	0x38(%rbx), %rsi
 39bcf84:      	testq	%rsi, %rsi
 39bcf87:      	je	0x39bcf98 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x38>
 39bcf89:      	movq	0x40(%rbx), %rdi
 39bcf8d:      	movl	$0x1, %edx
 39bcf92:      	callq	*0x1783da8(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf98:      	movq	0x58(%rbx), %rsi
 39bcf9c:      	testq	%rsi, %rsi
 39bcf9f:      	je	0x39bcfb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x50>
 39bcfa1:      	movq	0x60(%rbx), %rdi
 39bcfa5:      	movl	$0x1, %edx
 39bcfaa:      	callq	*0x1783d90(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcfb0:      	movq	0x70(%rbx), %rax
 39bcfb4:      	testq	%rax, %rax
 39bcfb7:      	je	0x39bcfc8 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x68>
 39bcfb9:      	lock
 39bcfba:      	decq	(%rax)
 39bcfbd:      	jne	0x39bcfc8 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x68>
 39bcfbf:      	leaq	0x70(%rbx), %rdi
 39bcfc3:      	callq	0x39b9fb0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 39bcfc8:      	movq	0xd8(%rbx), %r15
 39bcfcf:      	testq	%r15, %r15
 39bcfd2:      	je	0x39bcffc <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x9c>
 39bcfd4:      	movq	0xd0(%rbx), %r14
 39bcfdb:      	jmp	0x39bcfe9 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x89>
 39bcfdd:      	nopl	(%rax)
 39bcfe0:      	addq	$0x8, %r14
 39bcfe4:      	decq	%r15
 39bcfe7:      	je	0x39bcffc <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x9c>
 39bcfe9:      	movq	(%r14), %rax
 39bcfec:      	lock
 39bcfed:      	decq	(%rax)
 39bcff0:      	jne	0x39bcfe0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x80>
 39bcff2:      	movq	%r14, %rdi
 39bcff5:      	callq	0x39bcef0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 39bcffa:      	jmp	0x39bcfe0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x80>
 39bcffc:      	movq	0xc8(%rbx), %rsi
 39bd003:      	testq	%rsi, %rsi
 39bd006:      	je	0x39bd01e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0xbe>
 39bd008:      	movq	0xd0(%rbx), %rdi
 39bd00f:      	shlq	$0x3, %rsi
 39bd013:      	movl	$0x8, %edx
 39bd018:      	callq	*0x1783d22(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bd01e:      	cmpq	$-0x1, %rbx
 39bd022:      	je	0x39bd043 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0xe3>
 39bd024:      	lock
 39bd025:      	decq	0x8(%rbx)
 39bd029:      	jne	0x39bd043 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0xe3>
 39bd02b:      	movl	$0xe8, %esi
 39bd030:      	movl	$0x8, %edx
 39bd035:      	movq	%rbx, %rdi
 39bd038:      	popq	%rbx
 39bd039:      	popq	%r14
 39bd03b:      	popq	%r15
 39bd03d:      	jmpq	*0x1783cfd(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bd043:      	popq	%rbx
 39bd044:      	popq	%r14
 39bd046:      	popq	%r15
 39bd048:      	retq
 39bd049:      	movq	%rax, %r14
 39bd04c:      	leaq	0xb8(%rbx), %rdi
 39bd053:      	callq	0x3989580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockINtNtCscdodAO9FK5_5alloc3vec3VecINtNtB1x_4sync3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthEEEEB2k_>
 39bd058:      	cmpq	$-0x1, %rbx
 39bd05c:      	je	0x39bd078 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x118>
 39bd05e:      	lock
 39bd05f:      	decq	0x8(%rbx)
 39bd063:      	jne	0x39bd078 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x118>
 39bd065:      	movl	$0xe8, %esi
 39bd06a:      	movl	$0x8, %edx
 39bd06f:      	movq	%rbx, %rdi
 39bd072:      	callq	*0x1783cc8(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bd078:      	movq	%r14, %rdi
 39bd07b:      	callq	0x4fde6d0 <_Unwind_Resume@plt>

0000000003b1a500 <_RNvXs_NtNtCs2AWtUsOyxgP_3std4sync6poisonINtB4_11PoisonErrorINtNtB4_5mutex10MutexGuardINtNtCscdodAO9FK5_5alloc3vec3VecNtNtCs3pwlnhBXFtN_12memra_server5tests10MeterEventEEENtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmtB1V_>:
 3b1a500:      	pushq	%rbx
 3b1a501:      	subq	$0x10, %rsp
 3b1a505:      	leaq	-0x2d01248(%rip), %rdx  # 0xe192c4 <anon.0cae78d1abade22e61931222b273b913.50.llvm.5605875074211559972+0x396>
 3b1a50c:      	movq	%rsp, %rbx
 3b1a50f:      	movl	$0xb, %ecx
 3b1a514:      	movq	%rbx, %rdi
 3b1a517:      	callq	*0x16288a3(%rip)        # 0x5142dc0 <writev+0x5142dc0>
 3b1a51d:      	movq	%rbx, %rdi
 3b1a520:      	callq	*0x1628dfa(%rip)        # 0x5143320 <writev+0x5143320>
 3b1a526:      	addq	$0x10, %rsp
 3b1a52a:      	popq	%rbx
 3b1a52b:      	retq
 3b1a52c:      	int3
 3b1a52d:      	int3
 3b1a52e:      	int3
 3b1a52f:      	int3

0000000003b2b248 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>:
 3b2b248:      	movl	0x18(%rdi), %eax
 3b2b24b:      	testl	%eax, %eax
 3b2b24d:      	jne	0x3b2b250 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_+0x8>
 3b2b24f:      	retq
 3b2b250:      	subq	$0x28, %rsp
 3b2b254:      	leaq	0x18(%rsp), %rax
 3b2b259:      	movq	%rdi, (%rax)
 3b2b25c:      	addq	$0x18, %rdi
 3b2b260:      	leaq	0xf(%rsp), %rcx
 3b2b265:      	movq	%rcx, 0x8(%rax)
 3b2b269:      	leaq	0x10(%rsp), %rdx
 3b2b26e:      	movq	%rax, (%rdx)
 3b2b271:      	leaq	0x14fba50(%rip), %rcx   # 0x5026cc8 <anon.3222e37afe7119ca9a4f598626aaaddd.38.llvm.15726592429269621555>
 3b2b278:      	leaq	0x14fba71(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2b27f:      	pushq	$0x1
 3b2b281:      	popq	%rsi
 3b2b282:      	callq	*0x1615b18(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2b288:      	addq	$0x28, %rsp
 3b2b28c:      	retq

0000000003b2b3a1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>:
 3b2b3a1:      	movl	0x10(%rdi), %eax
 3b2b3a4:      	testl	%eax, %eax
 3b2b3a6:      	jne	0x3b2b3a9 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_+0x8>
 3b2b3a8:      	retq
 3b2b3a9:      	subq	$0x28, %rsp
 3b2b3ad:      	leaq	0x18(%rsp), %rax
 3b2b3b2:      	movq	%rdi, (%rax)
 3b2b3b5:      	addq	$0x10, %rdi
 3b2b3b9:      	leaq	0xf(%rsp), %rcx
 3b2b3be:      	movq	%rcx, 0x8(%rax)
 3b2b3c2:      	leaq	0x10(%rsp), %rdx
 3b2b3c7:      	movq	%rax, (%rdx)
 3b2b3ca:      	leaq	0x14fb9d7(%rip), %rcx   # 0x5026da8 <anon.3222e37afe7119ca9a4f598626aaaddd.46.llvm.15726592429269621555>
 3b2b3d1:      	leaq	0x14fb918(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2b3d8:      	pushq	$0x1
 3b2b3da:      	popq	%rsi
 3b2b3db:      	callq	*0x16159bf(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2b3e1:      	addq	$0x28, %rsp
 3b2b3e5:      	retq

0000000003b2b87b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>:
 3b2b87b:      	movl	0x10(%rdi), %eax
 3b2b87e:      	testl	%eax, %eax
 3b2b880:      	jne	0x3b2b883 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server+0x8>
 3b2b882:      	retq
 3b2b883:      	subq	$0x28, %rsp
 3b2b887:      	leaq	0x18(%rsp), %rax
 3b2b88c:      	movq	%rdi, (%rax)
 3b2b88f:      	addq	$0x10, %rdi
 3b2b893:      	leaq	0xf(%rsp), %rcx
 3b2b898:      	movq	%rcx, 0x8(%rax)
 3b2b89c:      	leaq	0x10(%rsp), %rdx
 3b2b8a1:      	movq	%rax, (%rdx)
 3b2b8a4:      	leaq	0x14fb7cd(%rip), %rcx   # 0x5027078 <anon.3222e37afe7119ca9a4f598626aaaddd.65.llvm.15726592429269621555>
 3b2b8ab:      	leaq	0x14fb43e(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2b8b2:      	pushq	$0x1
 3b2b8b4:      	popq	%rsi
 3b2b8b5:      	callq	*0x16154e5(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2b8bb:      	addq	$0x28, %rsp
 3b2b8bf:      	retq

0000000003b2c967 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>:
 3b2c967:      	movl	0x8(%rdi), %eax
 3b2c96a:      	testl	%eax, %eax
 3b2c96c:      	jne	0x3b2c96f <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_+0x8>
 3b2c96e:      	retq
 3b2c96f:      	subq	$0x28, %rsp
 3b2c973:      	leaq	0x18(%rsp), %rax
 3b2c978:      	movq	%rdi, (%rax)
 3b2c97b:      	addq	$0x8, %rdi
 3b2c97f:      	leaq	0xf(%rsp), %rcx
 3b2c984:      	movq	%rcx, 0x8(%rax)
 3b2c988:      	leaq	0x10(%rsp), %rdx
 3b2c98d:      	movq	%rax, (%rdx)
 3b2c990:      	leaq	0x14fb0b9(%rip), %rcx   # 0x5027a50 <anon.3222e37afe7119ca9a4f598626aaaddd.130.llvm.15726592429269621555>
 3b2c997:      	leaq	0x14fa352(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2c99e:      	pushq	$0x1
 3b2c9a0:      	popq	%rsi
 3b2c9a1:      	callq	*0x16143f9(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2c9a7:      	addq	$0x28, %rsp
 3b2c9ab:      	retq

0000000003b49e10 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 3b49e10:      	pushq	%r14
 3b49e12:      	pushq	%rbx
 3b49e13:      	pushq	%rax
 3b49e14:      	movq	%rdi, %rbx
 3b49e17:      	cmpb	$0x1, 0x31(%rdi)
 3b49e1b:      	jne	0x3b49e71 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3b49e1d:      	movq	0x8(%rbx), %rcx
 3b49e21:      	movq	(%rcx), %rax
 3b49e24:      	nopw	%cs:(%rax,%rax)
 3b49e30:      	testq	%rax, %rax
 3b49e33:      	je	0x3b49e40 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 3b49e35:      	leaq	-0x1(%rax), %rdx
 3b49e39:      	lock
 3b49e3a:      	cmpxchgq	%rdx, (%rcx)
 3b49e3e:      	jne	0x3b49e30 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 3b49e40:      	cmpb	$0x0, 0x32(%rbx)
 3b49e44:      	jne	0x3b49e71 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3b49e46:      	movq	(%rbx), %rcx
 3b49e49:      	movzbl	0x30(%rbx), %edx
 3b49e4d:      	movq	(%rcx,%rdx,8), %rax
 3b49e51:      	nopw	%cs:(%rax,%rax)
 3b49e60:      	testq	%rax, %rax
 3b49e63:      	je	0x3b49e71 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3b49e65:      	leaq	-0x1(%rax), %rsi
 3b49e69:      	lock
 3b49e6a:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 3b49e6f:      	jne	0x3b49e60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 3b49e71:      	cmpl	$-0x1, 0x28(%rbx)
 3b49e75:      	je	0x3b49eba <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3b49e77:      	movq	0x10(%rbx), %rdi
 3b49e7b:      	cmpq	$0x2, %rdi
 3b49e7f:      	ja	0x3b49ec2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 3b49e81:      	movq	0x18(%rbx), %rcx
 3b49e85:      	addq	$0x18, %rbx
 3b49e89:      	movq	0x30(%rcx,%rdi,8), %rax
 3b49e8e:      	nop
 3b49e90:      	testq	%rax, %rax
 3b49e93:      	je	0x3b49ea2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 3b49e95:      	leaq	-0x1(%rax), %rdx
 3b49e99:      	lock
 3b49e9a:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 3b49ea0:      	jne	0x3b49e90 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 3b49ea2:      	movq	(%rbx), %rax
 3b49ea5:      	lock
 3b49ea6:      	decq	(%rax)
 3b49ea9:      	jne	0x3b49eba <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3b49eab:      	movq	%rbx, %rdi
 3b49eae:      	addq	$0x8, %rsp
 3b49eb2:      	popq	%rbx
 3b49eb3:      	popq	%r14
 3b49eb5:      	jmp	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3b49eba:      	addq	$0x8, %rsp
 3b49ebe:      	popq	%rbx
 3b49ebf:      	popq	%r14
 3b49ec1:      	retq
 3b49ec2:      	leaq	0x14d8007(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 3b49ec9:      	movl	$0x3, %esi
 3b49ece:      	callq	*0x15f6fa4(%rip)        # 0x5140e78 <writev+0x5140e78>
 3b49ed4:      	ud2
 3b49ed6:      	movq	%rax, %r14
 3b49ed9:      	movq	0x18(%rbx), %rax
 3b49edd:      	lock
 3b49ede:      	decq	(%rax)
 3b49ee1:      	jne	0x3b49eef <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 3b49ee3:      	addq	$0x18, %rbx
 3b49ee7:      	movq	%rbx, %rdi
 3b49eea:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3b49eef:      	movq	%r14, %rdi
 3b49ef2:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3b49ef7:      	callq	*0x15f6ebb(%rip)        # 0x5140db8 <writev+0x5140db8>
 3b49efd:      	int3
 3b49efe:      	int3
 3b49eff:      	int3

0000000003b55f50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockAjj3_E10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0B2o_.llvm.15726592429269621555>:
 3b55f50:      	pushq	%r14
 3b55f52:      	pushq	%rbx
 3b55f53:      	subq	$0x48, %rsp
 3b55f57:      	movq	(%rdi), %rax
 3b55f5a:      	movq	(%rax), %r14
 3b55f5d:      	movq	$0x0, (%rax)
 3b55f64:      	testq	%r14, %r14
 3b55f67:      	je	0x3b55f8f <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockAjj3_E10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0B2o_.llvm.15726592429269621555+0x3f>
 3b55f69:      	callq	0x387b880 <_RNvNtCs3pwlnhBXFtN_12memra_server14route_contract22hybrid_interactive_cap>
 3b55f6e:      	movq	%rax, %rbx
 3b55f71:      	movq	%rsp, %rdi
 3b55f74:      	callq	*0x15ebc76(%rip)        # 0x5141bf0 <writev+0x5141bf0>
 3b55f7a:      	movups	0x28(%rsp), %xmm0
 3b55f7f:      	movq	%rbx, (%r14)
 3b55f82:      	movups	%xmm0, 0x8(%r14)
 3b55f87:      	addq	$0x48, %rsp
 3b55f8b:      	popq	%rbx
 3b55f8c:      	popq	%r14
 3b55f8e:      	retq
 3b55f8f:      	leaq	0x14d1fba(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b55f96:      	callq	*0x15eaea4(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b55f9c:      	int3
 3b55f9d:      	int3
 3b55f9e:      	int3
 3b55f9f:      	int3

0000000003b56360 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555>:
 3b56360:      	pushq	%r15
 3b56362:      	pushq	%r14
 3b56364:      	pushq	%rbx
 3b56365:      	subq	$0x20, %rsp
 3b56369:      	movq	(%rdi), %rax
 3b5636c:      	movq	(%rax), %r14
 3b5636f:      	movq	$0x0, (%rax)
 3b56376:      	testq	%r14, %r14
 3b56379:      	je	0x3b564ef <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x18f>
 3b5637f:      	leaq	-0x2d3c4b1(%rip), %rsi  # 0xe19ed5 <anon.3222e37afe7119ca9a4f598626aaaddd.276.llvm.15726592429269621555+0xae>
 3b56386:      	movq	%rsp, %rdi
 3b56389:      	movl	$0x15, %edx
 3b5638e:      	callq	*0x15eaab4(%rip)        # 0x5140e48 <writev+0x5140e48>
 3b56394:      	cmpl	$0x1, (%rsp)
 3b56398:      	jne	0x3b563bc <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x5c>
 3b5639a:      	movq	0x8(%rsp), %rsi
 3b5639f:      	cmpq	$-0x1, %rsi
 3b563a3:      	je	0x3b563ee <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563a5:      	testq	%rsi, %rsi
 3b563a8:      	je	0x3b563ee <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563aa:      	movq	0x10(%rsp), %rdi
 3b563af:      	movl	$0x1, %edx
 3b563b4:      	callq	*0x15ea986(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b563ba:      	jmp	0x3b563ee <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563bc:      	movq	0x8(%rsp), %rsi
 3b563c1:      	cmpq	$-0x1, %rsi
 3b563c5:      	je	0x3b563ee <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563c7:      	movq	0x10(%rsp), %rdi
 3b563cc:      	movq	0x18(%rsp), %rcx
 3b563d1:      	testq	%rcx, %rcx
 3b563d4:      	je	0x3b563f3 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x93>
 3b563d6:      	cmpq	$0x1, %rcx
 3b563da:      	jne	0x3b56417 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xb7>
 3b563dc:      	movzbl	(%rdi), %eax
 3b563df:      	xorl	%r15d, %r15d
 3b563e2:      	cmpl	$0x2b, %eax
 3b563e5:      	je	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b563e7:      	cmpl	$0x2d, %eax
 3b563ea:      	je	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b563ec:      	jmp	0x3b5641a <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xba>
 3b563ee:      	xorl	%r15d, %r15d
 3b563f1:      	jmp	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xa6>
 3b563f3:      	movq	%rcx, %r15
 3b563f6:      	testq	%rsi, %rsi
 3b563f9:      	je	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xa6>
 3b563fb:      	movl	$0x1, %edx
 3b56400:      	callq	*0x15ea93a(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b56406:      	movq	%r15, (%r14)
 3b56409:      	movq	%rbx, 0x8(%r14)
 3b5640d:      	addq	$0x20, %rsp
 3b56411:      	popq	%rbx
 3b56412:      	popq	%r14
 3b56414:      	popq	%r15
 3b56416:      	retq
 3b56417:      	movzbl	(%rdi), %eax
 3b5641a:      	xorl	%r8d, %r8d
 3b5641d:      	cmpb	$0x2b, %al
 3b5641f:      	sete	%r8b
 3b56423:      	movq	%r8, %rax
 3b56426:      	negq	%rax
 3b56429:      	movq	%rcx, %rdx
 3b5642c:      	subq	%r8, %rdx
 3b5642f:      	addq	%rdi, %r8
 3b56432:      	cmpq	$0x11, %rdx
 3b56436:      	jae	0x3b56487 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x127>
 3b56438:      	movl	$0x1, %r15d
 3b5643e:      	testq	%rdx, %rdx
 3b56441:      	je	0x3b564d5 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x175>
 3b56447:      	addq	%rax, %rcx
 3b5644a:      	negq	%rcx
 3b5644d:      	xorl	%eax, %eax
 3b5644f:      	xorl	%ebx, %ebx
 3b56451:      	nopw	%cs:(%rax,%rax)
 3b56460:      	movzbl	(%r8,%rax), %edx
 3b56465:      	addl	$-0x30, %edx
 3b56468:      	cmpl	$0x9, %edx
 3b5646b:      	ja	0x3b564dc <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x17c>
 3b5646d:      	leaq	(%rbx,%rbx,4), %r9
 3b56471:      	movl	%edx, %edx
 3b56473:      	leaq	(%rdx,%r9,2), %rbx
 3b56477:      	incq	%rax
 3b5647a:      	movq	%rcx, %rdx
 3b5647d:      	addq	%rax, %rdx
 3b56480:      	jne	0x3b56460 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x100>
 3b56482:      	jmp	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b56487:      	addq	%rax, %rcx
 3b5648a:      	negq	%rcx
 3b5648d:      	xorl	%r15d, %r15d
 3b56490:      	movl	$0xa, %r9d
 3b56496:      	xorl	%r10d, %r10d
 3b56499:      	xorl	%ebx, %ebx
 3b5649b:      	movq	%rcx, %rax
 3b5649e:      	addq	%r10, %rax
 3b564a1:      	je	0x3b564e4 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x184>
 3b564a3:      	movq	%rbx, %rax
 3b564a6:      	mulq	%r9
 3b564a9:      	jo	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564af:      	movq	%rax, %rbx
 3b564b2:      	movzbl	(%r8,%r10), %edx
 3b564b7:      	addl	$-0x30, %edx
 3b564ba:      	addq	%rdx, %rbx
 3b564bd:      	setb	%al
 3b564c0:      	cmpl	$0x9, %edx
 3b564c3:      	ja	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564c9:      	incq	%r10
 3b564cc:      	testb	%al, %al
 3b564ce:      	je	0x3b5649b <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x13b>
 3b564d0:      	jmp	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564d5:      	xorl	%ebx, %ebx
 3b564d7:      	jmp	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564dc:      	xorl	%r15d, %r15d
 3b564df:      	jmp	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564e4:      	movl	$0x1, %r15d
 3b564ea:      	jmp	0x3b563f6 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564ef:      	leaq	0x14d1a5a(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b564f6:      	callq	*0x15ea944(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b564fc:      	int3
 3b564fd:      	int3
 3b564fe:      	int3
 3b564ff:      	int3

0000000003b58580 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockNtNtBc_4time7InstantE10initializeNCINvB1a_11get_or_initNvMB1I_B1G_3nowE0zE0E0Cs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555>:
 3b58580:      	pushq	%rbx
 3b58581:      	movq	(%rdi), %rax
 3b58584:      	movq	(%rax), %rbx
 3b58587:      	movq	$0x0, (%rax)
 3b5858e:      	testq	%rbx, %rbx
 3b58591:      	je	0x3b585a1 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockNtNtBc_4time7InstantE10initializeNCINvB1a_11get_or_initNvMB1I_B1G_3nowE0zE0E0Cs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555+0x21>
 3b58593:      	callq	*0x15e8917(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3b58599:      	movq	%rax, (%rbx)
 3b5859c:      	movl	%edx, 0x8(%rbx)
 3b5859f:      	popq	%rbx
 3b585a0:      	retq
 3b585a1:      	leaq	0x14cf9a8(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b585a8:      	callq	*0x15e8892(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b585ae:      	int3
 3b585af:      	int3

0000000003b5cde0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555>:
 3b5cde0:      	pushq	%r14
 3b5cde2:      	pushq	%rbx
 3b5cde3:      	subq	$0x28, %rsp
 3b5cde7:      	movq	(%rdi), %rax
 3b5cdea:      	movq	(%rax), %rbx
 3b5cded:      	movq	$0x0, (%rax)
 3b5cdf4:      	testq	%rbx, %rbx
 3b5cdf7:      	je	0x3b5cf65 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x185>
 3b5cdfd:      	leaq	-0x2d42eaa(%rip), %rsi  # 0xe19f5a <anon.3222e37afe7119ca9a4f598626aaaddd.303.llvm.15726592429269621555+0x13>
 3b5ce04:      	leaq	0x8(%rsp), %rdi
 3b5ce09:      	movl	$0x1a, %edx
 3b5ce0e:      	callq	*0x15e4034(%rip)        # 0x5140e48 <writev+0x5140e48>
 3b5ce14:      	cmpl	$0x1, 0x8(%rsp)
 3b5ce19:      	jne	0x3b5ce3d <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x5d>
 3b5ce1b:      	movq	0x10(%rsp), %rsi
 3b5ce20:      	cmpq	$-0x1, %rsi
 3b5ce24:      	je	0x3b5ce77 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce26:      	testq	%rsi, %rsi
 3b5ce29:      	je	0x3b5ce77 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce2b:      	movq	0x18(%rsp), %rdi
 3b5ce30:      	movl	$0x1, %edx
 3b5ce35:      	callq	*0x15e3f05(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b5ce3b:      	jmp	0x3b5ce77 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce3d:      	movq	0x10(%rsp), %rsi
 3b5ce42:      	cmpq	$-0x1, %rsi
 3b5ce46:      	je	0x3b5ce77 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce48:      	movq	0x18(%rsp), %rdi
 3b5ce4d:      	movq	0x20(%rsp), %rcx
 3b5ce52:      	testq	%rcx, %rcx
 3b5ce55:      	je	0x3b5ce7f <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x9f>
 3b5ce57:      	cmpq	$0x1, %rcx
 3b5ce5b:      	jne	0x3b5ce90 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0xb0>
 3b5ce5d:      	movzbl	(%rdi), %eax
 3b5ce60:      	xorl	%r14d, %r14d
 3b5ce63:      	cmpl	$0x2b, %eax
 3b5ce66:      	je	0x3b5cf40 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5ce6c:      	cmpl	$0x2d, %eax
 3b5ce6f:      	je	0x3b5cf40 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5ce75:      	jmp	0x3b5ce93 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0xb3>
 3b5ce77:      	xorl	%r14d, %r14d
 3b5ce7a:      	jmp	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5ce7f:      	movq	%rcx, %r14
 3b5ce82:      	testq	%rsi, %rsi
 3b5ce85:      	jne	0x3b5cf45 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x165>
 3b5ce8b:      	jmp	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5ce90:      	movzbl	(%rdi), %eax
 3b5ce93:      	xorl	%r8d, %r8d
 3b5ce96:      	cmpb	$0x2b, %al
 3b5ce98:      	sete	%r8b
 3b5ce9c:      	movq	%r8, %rax
 3b5ce9f:      	negq	%rax
 3b5cea2:      	movq	%rcx, %rdx
 3b5cea5:      	subq	%r8, %rdx
 3b5cea8:      	addq	%rdi, %r8
 3b5ceab:      	cmpq	$0x11, %rdx
 3b5ceaf:      	jae	0x3b5cef4 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x114>
 3b5ceb1:      	testq	%rdx, %rdx
 3b5ceb4:      	je	0x3b5cf33 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x153>
 3b5ceb6:      	addq	%rax, %rcx
 3b5ceb9:      	negq	%rcx
 3b5cebc:      	xorl	%eax, %eax
 3b5cebe:      	xorl	%r14d, %r14d
 3b5cec1:      	nopw	%cs:(%rax,%rax)
 3b5ced0:      	movzbl	(%r8,%rax), %edx
 3b5ced5:      	addl	$-0x30, %edx
 3b5ced8:      	cmpl	$0x9, %edx
 3b5cedb:      	ja	0x3b5cf3d <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x15d>
 3b5cedd:      	leaq	(%r14,%r14,4), %r9
 3b5cee1:      	movl	%edx, %edx
 3b5cee3:      	leaq	(%rdx,%r9,2), %r14
 3b5cee7:      	incq	%rax
 3b5ceea:      	movq	%rcx, %rdx
 3b5ceed:      	addq	%rax, %rdx
 3b5cef0:      	jne	0x3b5ced0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0xf0>
 3b5cef2:      	jmp	0x3b5cf40 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cef4:      	addq	%rax, %rcx
 3b5cef7:      	negq	%rcx
 3b5cefa:      	xorl	%r14d, %r14d
 3b5cefd:      	movl	$0xa, %r9d
 3b5cf03:      	xorl	%r10d, %r10d
 3b5cf06:      	xorl	%eax, %eax
 3b5cf08:      	movq	%rcx, %rdx
 3b5cf0b:      	addq	%r10, %rdx
 3b5cf0e:      	je	0x3b5cf5b <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x17b>
 3b5cf10:      	mulq	%r9
 3b5cf13:      	jo	0x3b5cf40 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf15:      	movzbl	(%r8,%r10), %r11d
 3b5cf1a:      	addl	$-0x30, %r11d
 3b5cf1e:      	addq	%r11, %rax
 3b5cf21:      	setb	%dl
 3b5cf24:      	cmpl	$0x9, %r11d
 3b5cf28:      	ja	0x3b5cf40 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf2a:      	incq	%r10
 3b5cf2d:      	testb	%dl, %dl
 3b5cf2f:      	je	0x3b5cf08 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x128>
 3b5cf31:      	jmp	0x3b5cf40 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf33:      	xorl	%r14d, %r14d
 3b5cf36:      	testq	%rsi, %rsi
 3b5cf39:      	jne	0x3b5cf45 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x165>
 3b5cf3b:      	jmp	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cf3d:      	xorl	%r14d, %r14d
 3b5cf40:      	testq	%rsi, %rsi
 3b5cf43:      	je	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cf45:      	movl	$0x1, %edx
 3b5cf4a:      	callq	*0x15e3df0(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b5cf50:      	movq	%r14, (%rbx)
 3b5cf53:      	addq	$0x28, %rsp
 3b5cf57:      	popq	%rbx
 3b5cf58:      	popq	%r14
 3b5cf5a:      	retq
 3b5cf5b:      	movq	%rax, %r14
 3b5cf5e:      	testq	%rsi, %rsi
 3b5cf61:      	jne	0x3b5cf45 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x165>
 3b5cf63:      	jmp	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cf65:      	leaq	0x14cafe4(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b5cf6c:      	callq	*0x15e3ece(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b5cf72:      	int3
 3b5cf73:      	int3
 3b5cf74:      	int3
 3b5cf75:      	int3
 3b5cf76:      	int3
 3b5cf77:      	int3
 3b5cf78:      	int3
 3b5cf79:      	int3
 3b5cf7a:      	int3
 3b5cf7b:      	int3
 3b5cf7c:      	int3
 3b5cf7d:      	int3
 3b5cf7e:      	int3
 3b5cf7f:      	int3

0000000003b83290 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockAjj3_E10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB2t_.llvm.15726592429269621555>:
 3b83290:      	pushq	%r14
 3b83292:      	pushq	%rbx
 3b83293:      	subq	$0x48, %rsp
 3b83297:      	movq	(%rdi), %rax
 3b8329a:      	movq	(%rax), %r14
 3b8329d:      	movq	$0x0, (%rax)
 3b832a4:      	testq	%r14, %r14
 3b832a7:      	je	0x3b832cf <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockAjj3_E10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB2t_.llvm.15726592429269621555+0x3f>
 3b832a9:      	callq	0x387b880 <_RNvNtCs3pwlnhBXFtN_12memra_server14route_contract22hybrid_interactive_cap>
 3b832ae:      	movq	%rax, %rbx
 3b832b1:      	movq	%rsp, %rdi
 3b832b4:      	callq	*0x15be936(%rip)        # 0x5141bf0 <writev+0x5141bf0>
 3b832ba:      	movups	0x28(%rsp), %xmm0
 3b832bf:      	movq	%rbx, (%r14)
 3b832c2:      	movups	%xmm0, 0x8(%r14)
 3b832c7:      	addq	$0x48, %rsp
 3b832cb:      	popq	%rbx
 3b832cc:      	popq	%r14
 3b832ce:      	retq
 3b832cf:      	leaq	0x14a4c7a(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b832d6:      	callq	*0x15bdb64(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b832dc:      	int3
 3b832dd:      	int3
 3b832de:      	int3
 3b832df:      	int3

0000000003b83470 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0INtNtNtB1Q_3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB32_.llvm.15726592429269621555>:
 3b83470:      	pushq	%rax
 3b83471:      	movq	(%rdi), %rax
 3b83474:      	movq	%rax, (%rsp)
 3b83478:      	movq	%rsp, %rdi
 3b8347b:      	callq	0x3b56360 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555>
 3b83480:      	popq	%rax
 3b83481:      	retq
 3b83482:      	int3
 3b83483:      	int3
 3b83484:      	int3
 3b83485:      	int3
 3b83486:      	int3
 3b83487:      	int3
 3b83488:      	int3
 3b83489:      	int3
 3b8348a:      	int3
 3b8348b:      	int3
 3b8348c:      	int3
 3b8348d:      	int3
 3b8348e:      	int3
 3b8348f:      	int3

0000000003b838a0 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockNtNtBh_4time7InstantE10initializeNCINvB1f_11get_or_initNvMB1N_B1L_3nowE0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableCs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555>:
 3b838a0:      	pushq	%rbx
 3b838a1:      	movq	(%rdi), %rax
 3b838a4:      	movq	(%rax), %rbx
 3b838a7:      	movq	$0x0, (%rax)
 3b838ae:      	testq	%rbx, %rbx
 3b838b1:      	je	0x3b838c1 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockNtNtBh_4time7InstantE10initializeNCINvB1f_11get_or_initNvMB1N_B1L_3nowE0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableCs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555+0x21>
 3b838b3:      	callq	*0x15bd5f7(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3b838b9:      	movq	%rax, (%rbx)
 3b838bc:      	movl	%edx, 0x8(%rbx)
 3b838bf:      	popq	%rbx
 3b838c0:      	retq
 3b838c1:      	leaq	0x14a4688(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b838c8:      	callq	*0x15bd572(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b838ce:      	int3
 3b838cf:      	int3

0000000003b85730 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockyE10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB2p_.llvm.15726592429269621555>:
 3b85730:      	pushq	%rax
 3b85731:      	movq	(%rdi), %rax
 3b85734:      	movq	%rax, (%rsp)
 3b85738:      	movq	%rsp, %rdi
 3b8573b:      	callq	0x3b5cde0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555>
 3b85740:      	popq	%rax
 3b85741:      	retq
 3b85742:      	int3
 3b85743:      	int3
 3b85744:      	int3
 3b85745:      	int3
 3b85746:      	int3
 3b85747:      	int3
 3b85748:      	int3
 3b85749:      	int3
 3b8574a:      	int3
 3b8574b:      	int3
 3b8574c:      	int3
 3b8574d:      	int3
 3b8574e:      	int3
 3b8574f:      	int3

0000000003c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEEB1T_.llvm.12140772379724168186>:
 3c03360:      	pushq	%rbx
 3c03361:      	movq	(%rdi), %rbx
 3c03364:      	cmpb	$0x0, 0x8(%rdi)
 3c03368:      	jne	0x3c0337c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x1c>
 3c0336a:      	movq	0x153dcf7(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c03371:      	movq	(%rax), %rax
 3c03374:      	shlq	%rax
 3c03377:      	testq	%rax, %rax
 3c0337a:      	jne	0x3c03391 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x31>
 3c0337c:      	xorl	%eax, %eax
 3c0337e:      	xchgl	%eax, (%rbx)
 3c03380:      	cmpl	$0x2, %eax
 3c03383:      	je	0x3c03387 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x27>
 3c03385:      	popq	%rbx
 3c03386:      	retq
 3c03387:      	movq	%rbx, %rdi
 3c0338a:      	popq	%rbx
 3c0338b:      	jmpq	*0x153dcdf(%rip)        # 0x5141070 <writev+0x5141070>
 3c03391:      	callq	*0x153dce1(%rip)        # 0x5141078 <writev+0x5141078>
 3c03397:      	testb	%al, %al
 3c03399:      	jne	0x3c0337c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x1c>
 3c0339b:      	movb	$0x1, 0x4(%rbx)
 3c0339f:      	jmp	0x3c0337c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x1c>
 3c033a1:      	int3
 3c033a2:      	int3
 3c033a3:      	int3
 3c033a4:      	int3
 3c033a5:      	int3
 3c033a6:      	int3
 3c033a7:      	int3
 3c033a8:      	int3
 3c033a9:      	int3
 3c033aa:      	int3
 3c033ab:      	int3
 3c033ac:      	int3
 3c033ad:      	int3
 3c033ae:      	int3
 3c033af:      	int3

0000000003c736a0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output>:
 3c736a0:      	pushq	%rbp
 3c736a1:      	pushq	%r15
 3c736a3:      	pushq	%r14
 3c736a5:      	pushq	%r13
 3c736a7:      	pushq	%r12
 3c736a9:      	pushq	%rbx
 3c736aa:      	subq	$0x78, %rsp
 3c736ae:      	movq	%rcx, %r14
 3c736b1:      	movq	%rdx, %r15
 3c736b4:      	movq	%rsi, %r13
 3c736b7:      	movq	%rdi, %rbx
 3c736ba:      	movl	$0x1, %ecx
 3c736bf:      	xorl	%eax, %eax
 3c736c1:      	lock
 3c736c2:      	cmpxchgl	%ecx, (%rdi)
 3c736c5:      	jne	0x3c739ff <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x35f>
 3c736cb:      	movq	0x14cd996(%rip), %r12   # 0x5141068 <writev+0x5141068>
 3c736d2:      	movq	(%r12), %rax
 3c736d6:      	shlq	%rax
 3c736d9:      	testq	%rax, %rax
 3c736dc:      	jne	0x3c73a0d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x36d>
 3c736e2:      	xorl	%esi, %esi
 3c736e4:      	movzbl	0x4(%rbx), %eax
 3c736e8:      	testb	%al, %al
 3c736ea:      	jne	0x3c73a25 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x385>
 3c736f0:      	cmpq	$0x0, 0x20(%rbx)
 3c736f5:      	je	0x3c7395b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bb>
 3c736fb:      	movl	%esi, 0xc(%rsp)
 3c736ff:      	leaq	0x28(%rbx), %rdi
 3c73703:      	movq	%r13, 0x28(%rsp)
 3c73708:      	movq	%r13, %rsi
 3c7370b:      	movq	%r15, %rdx
 3c7370e:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73713:      	movq	%rax, %r13
 3c73716:      	shrq	$0x39, %rax
 3c7371a:      	movq	0x8(%rbx), %rdx
 3c7371e:      	movq	0x10(%rbx), %rcx
 3c73722:      	movd	%eax, %xmm0
 3c73726:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c7372a:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c7372f:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73734:      	xorl	%esi, %esi
 3c73736:      	pcmpeqd	%xmm2, %xmm2
 3c7373a:      	movq	0x14cd667(%rip), %r8    # 0x5140da8 <writev+0x5140da8>
 3c73741:      	andq	%rcx, %r13
 3c73744:      	movdqu	(%rdx,%r13), %xmm3
 3c7374a:      	movdqa	%xmm3, %xmm0
 3c7374e:      	pcmpeqb	%xmm1, %xmm0
 3c73752:      	pmovmskb	%xmm0, %r12d
 3c73757:      	testl	%r12d, %r12d
 3c7375a:      	je	0x3c737f0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x150>
 3c73760:      	movq	%r14, 0x10(%rsp)
 3c73765:      	movq	%rdx, 0x18(%rsp)
 3c7376a:      	movdqa	%xmm1, 0x50(%rsp)
 3c73770:      	movq	%rsi, 0x40(%rsp)
 3c73775:      	movdqa	%xmm3, 0x30(%rsp)
 3c7377b:      	tzcntl	%r12d, %eax
 3c73780:      	addq	%r13, %rax
 3c73783:      	andq	%rcx, %rax
 3c73786:      	shlq	$0x8, %rax
 3c7378a:      	movq	%rdx, %r14
 3c7378d:      	subq	%rax, %r14
 3c73790:      	cmpq	-0xf0(%r14), %r15
 3c73797:      	jne	0x3c737bf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x11f>
 3c73799:      	movq	-0xf8(%r14), %rsi
 3c737a0:      	movq	0x28(%rsp), %rdi
 3c737a5:      	movq	%r15, %rdx
 3c737a8:      	movq	%rcx, 0x20(%rsp)
 3c737ad:      	movq	%r8, %rbp
 3c737b0:      	callq	*%r8
 3c737b3:      	movq	%rbp, %r8
 3c737b6:      	movq	0x20(%rsp), %rcx
 3c737bb:      	testl	%eax, %eax
 3c737bd:      	je	0x3c73810 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x170>
 3c737bf:      	leal	-0x1(%r12), %eax
 3c737c4:      	andw	%r12w, %ax
 3c737c8:      	movl	%eax, %r12d
 3c737cb:      	movq	0x10(%rsp), %r14
 3c737d0:      	movq	0x18(%rsp), %rdx
 3c737d5:      	movdqa	0x50(%rsp), %xmm1
 3c737db:      	movq	0x40(%rsp), %rsi
 3c737e0:      	pcmpeqd	%xmm2, %xmm2
 3c737e4:      	movdqa	0x30(%rsp), %xmm3
 3c737ea:      	jne	0x3c7377b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0xdb>
 3c737ec:      	nopl	(%rax)
 3c737f0:      	pcmpeqb	%xmm2, %xmm3
 3c737f4:      	pmovmskb	%xmm3, %eax
 3c737f8:      	testl	%eax, %eax
 3c737fa:      	jne	0x3c73964 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2c4>
 3c73800:      	addq	%rsi, %r13
 3c73803:      	addq	$0x10, %r13
 3c73807:      	addq	$0x10, %rsi
 3c7380b:      	jmp	0x3c73741 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0xa1>
 3c73810:      	movb	$0x1, %al
 3c73812:      	cmpb	$0x2, -0x88(%r14)
 3c7381a:      	jae	0x3c73966 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2c6>
 3c73820:      	cmpq	$-0x2, -0x80(%r14)
 3c73825:      	movq	0x14cd83c(%rip), %r12   # 0x5141068 <writev+0x5141068>
 3c7382c:      	movl	0xc(%rsp), %esi
 3c73830:      	jne	0x3c7395d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bd>
 3c73836:      	movq	-0x8(%r14), %rax
 3c7383a:      	movq	0x10(%rsp), %rdi
 3c7383f:      	cmpq	%rax, %rdi
 3c73842:      	cmovbeq	%rax, %rdi
 3c73846:      	movq	0x38(%rbx), %rdx
 3c7384a:      	xorl	%ecx, %ecx
 3c7384c:      	subq	%rax, %rdx
 3c7384f:      	cmovaeq	%rdx, %rcx
 3c73853:      	movb	$0x2, %al
 3c73855:      	movq	%rdi, 0x10(%rsp)
 3c7385a:      	addq	%rdi, %rcx
 3c7385d:      	jb	0x3c7395d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bd>
 3c73863:      	cmpq	0x40(%rbx), %rcx
 3c73867:      	ja	0x3c7395d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bd>
 3c7386d:      	movq	%r15, %r13
 3c73870:      	movq	%rcx, 0x38(%rbx)
 3c73874:      	leaq	0x28(%rbx), %rdi
 3c73878:      	movq	0x28(%rsp), %rsi
 3c7387d:      	movq	%r15, %rdx
 3c73880:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73885:      	movq	%rax, %rcx
 3c73888:      	shrq	$0x39, %rcx
 3c7388c:      	movd	%ecx, %xmm0
 3c73890:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73894:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73899:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c7389e:      	xorl	%r15d, %r15d
 3c738a1:      	pcmpeqd	%xmm2, %xmm2
 3c738a5:      	movq	0x20(%rsp), %rdx
 3c738aa:      	andq	%rdx, %rax
 3c738ad:      	movq	0x18(%rsp), %rcx
 3c738b2:      	movdqu	(%rcx,%rax), %xmm3
 3c738b7:      	movdqa	%xmm3, %xmm0
 3c738bb:      	pcmpeqb	%xmm1, %xmm0
 3c738bf:      	pmovmskb	%xmm0, %r12d
 3c738c4:      	testl	%r12d, %r12d
 3c738c7:      	je	0x3c7393b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x29b>
 3c738c9:      	movq	%rax, 0x50(%rsp)
 3c738ce:      	movdqa	%xmm1, 0x40(%rsp)
 3c738d4:      	movdqa	%xmm3, 0x30(%rsp)
 3c738da:      	tzcntl	%r12d, %ecx
 3c738df:      	addq	%rax, %rcx
 3c738e2:      	andq	%rdx, %rcx
 3c738e5:      	shlq	$0x8, %rcx
 3c738e9:      	movq	0x18(%rsp), %r14
 3c738ee:      	subq	%rcx, %r14
 3c738f1:      	cmpq	-0xf0(%r14), %r13
 3c738f8:      	jne	0x3c73913 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x273>
 3c738fa:      	movq	%r13, %rdx
 3c738fd:      	movq	-0xf8(%r14), %rsi
 3c73904:      	movq	0x28(%rsp), %rdi
 3c73909:      	callq	*%rbp
 3c7390b:      	testl	%eax, %eax
 3c7390d:      	je	0x3c739a5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x305>
 3c73913:      	leal	-0x1(%r12), %eax
 3c73918:      	andw	%r12w, %ax
 3c7391c:      	movl	%eax, %r12d
 3c7391f:      	movq	0x50(%rsp), %rax
 3c73924:      	movq	0x20(%rsp), %rdx
 3c73929:      	movdqa	0x40(%rsp), %xmm1
 3c7392f:      	pcmpeqd	%xmm2, %xmm2
 3c73933:      	movdqa	0x30(%rsp), %xmm3
 3c73939:      	jne	0x3c738da <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x23a>
 3c7393b:      	pcmpeqb	%xmm2, %xmm3
 3c7393f:      	pmovmskb	%xmm3, %ecx
 3c73943:      	testl	%ecx, %ecx
 3c73945:      	jne	0x3c739ec <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x34c>
 3c7394b:      	addq	%r15, %rax
 3c7394e:      	addq	$0x10, %rax
 3c73952:      	addq	$0x10, %r15
 3c73956:      	jmp	0x3c738a5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x205>
 3c7395b:      	xorl	%eax, %eax
 3c7395d:      	testb	%sil, %sil
 3c73960:      	je	0x3c73976 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2d6>
 3c73962:      	jmp	0x3c7398d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73964:      	xorl	%eax, %eax
 3c73966:      	movq	0x14cd6fb(%rip), %r12   # 0x5141068 <writev+0x5141068>
 3c7396d:      	movl	0xc(%rsp), %esi
 3c73971:      	testb	%sil, %sil
 3c73974:      	jne	0x3c7398d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73976:      	movq	(%r12), %rcx
 3c7397a:      	movabsq	$0x7fffffffffffffff, %rdx # imm = 0x7FFFFFFFFFFFFFFF
 3c73984:      	testq	%rdx, %rcx
 3c73987:      	jne	0x3c73a56 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x3b6>
 3c7398d:      	xorl	%ecx, %ecx
 3c7398f:      	xchgl	%ecx, (%rbx)
 3c73991:      	cmpl	$0x2, %ecx
 3c73994:      	je	0x3c739dd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x33d>
 3c73996:      	addq	$0x78, %rsp
 3c7399a:      	popq	%rbx
 3c7399b:      	popq	%r12
 3c7399d:      	popq	%r13
 3c7399f:      	popq	%r14
 3c739a1:      	popq	%r15
 3c739a3:      	popq	%rbp
 3c739a4:      	retq
 3c739a5:      	movq	0x10(%rsp), %rax
 3c739aa:      	movq	%rax, -0x8(%r14)
 3c739ae:      	cmpb	$0x0, 0xc(%rsp)
 3c739b3:      	jne	0x3c739d2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x332>
 3c739b5:      	movq	0x14cd6ac(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c739bc:      	movq	(%rax), %rax
 3c739bf:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c739c9:      	testq	%rcx, %rax
 3c739cc:      	jne	0x3c73a73 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x3d3>
 3c739d2:      	xorl	%ecx, %ecx
 3c739d4:      	xchgl	%ecx, (%rbx)
 3c739d6:      	movb	$-0x1, %al
 3c739d8:      	cmpl	$0x2, %ecx
 3c739db:      	jne	0x3c73996 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2f6>
 3c739dd:      	movq	%rbx, %rdi
 3c739e0:      	movl	%eax, %ebx
 3c739e2:      	callq	*0x14cd688(%rip)        # 0x5141070 <writev+0x5141070>
 3c739e8:      	movl	%ebx, %eax
 3c739ea:      	jmp	0x3c73996 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2f6>
 3c739ec:      	leaq	0x13b901d(%rip), %rdi   # 0x502ca10 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x250>
 3c739f3:      	movl	0xc(%rsp), %ebp
 3c739f7:      	callq	*0x14cd443(%rip)        # 0x5140e40 <writev+0x5140e40>
 3c739fd:      	jmp	0x3c73a54 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x3b4>
 3c739ff:      	movq	%rbx, %rdi
 3c73a02:      	callq	*0x14cda00(%rip)        # 0x5141408 <writev+0x5141408>
 3c73a08:      	jmp	0x3c736cb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2b>
 3c73a0d:      	callq	*0x14cd665(%rip)        # 0x5141078 <writev+0x5141078>
 3c73a13:      	movl	%eax, %esi
 3c73a15:      	xorb	$0x1, %sil
 3c73a19:      	movzbl	0x4(%rbx), %eax
 3c73a1d:      	testb	%al, %al
 3c73a1f:      	je	0x3c736f0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x50>
 3c73a25:      	movq	%rbx, 0x68(%rsp)
 3c73a2a:      	movb	%sil, 0x70(%rsp)
 3c73a2f:      	leaq	-0x2e55055(%rip), %rdi  # 0xe1e9e1 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c73a36:      	leaq	0x13b8ab3(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c73a3d:      	leaq	0x13b8fb4(%rip), %r8    # 0x502c9f8 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x238>
 3c73a44:      	leaq	0x68(%rsp), %rdx
 3c73a49:      	movl	$0x2b, %esi
 3c73a4e:      	callq	*0x14cd444(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c73a54:      	ud2
 3c73a56:      	movl	%eax, %ebp
 3c73a58:      	callq	*0x14cd61a(%rip)        # 0x5141078 <writev+0x5141078>
 3c73a5e:      	movl	%eax, %ecx
 3c73a60:      	movl	%ebp, %eax
 3c73a62:      	testb	%cl, %cl
 3c73a64:      	jne	0x3c7398d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73a6a:      	movb	$0x1, 0x4(%rbx)
 3c73a6e:      	jmp	0x3c7398d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73a73:      	callq	*0x14cd5ff(%rip)        # 0x5141078 <writev+0x5141078>
 3c73a79:      	testb	%al, %al
 3c73a7b:      	jne	0x3c739d2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x332>
 3c73a81:      	movb	$0x1, 0x4(%rbx)
 3c73a85:      	jmp	0x3c739d2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x332>
 3c73a8a:      	movq	%rax, %r14
 3c73a8d:      	leaq	0x68(%rsp), %rdi
 3c73a92:      	callq	0x3c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c73a97:      	jmp	0x3c73aae <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x40e>
 3c73a99:      	callq	*0x14cd319(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c73a9f:      	movq	%rax, %r14
 3c73aa2:      	movzbl	%bpl, %esi
 3c73aa6:      	movq	%rbx, %rdi
 3c73aa9:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c73aae:      	movq	%r14, %rdi
 3c73ab1:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c73ab6:      	callq	*0x14cd2fc(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c73abc:      	int3
 3c73abd:      	int3
 3c73abe:      	int3
 3c73abf:      	int3

0000000003c73ac0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal>:
 3c73ac0:      	pushq	%rbp
 3c73ac1:      	pushq	%r15
 3c73ac3:      	pushq	%r14
 3c73ac5:      	pushq	%r13
 3c73ac7:      	pushq	%r12
 3c73ac9:      	pushq	%rbx
 3c73aca:      	subq	$0x128, %rsp            # imm = 0x128
 3c73ad1:      	movq	%r9, 0x18(%rsp)
 3c73ad6:      	movq	%r8, 0x38(%rsp)
 3c73adb:      	movq	%rcx, %r13
 3c73ade:      	movb	$0x3, %r14b
 3c73ae1:      	cmpb	$0x2, 0x60(%rcx)
 3c73ae5:      	jb	0x3c73dec <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x32c>
 3c73aeb:      	movq	%rdx, %r15
 3c73aee:      	movq	%rsi, %r12
 3c73af1:      	movq	%rdi, %rbx
 3c73af4:      	movq	%r13, %rdi
 3c73af7:      	callq	0x3c6efb0 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store17record_size_bytes>
 3c73afc:      	movq	%rax, %rbp
 3c73aff:      	movl	$0x1, %ecx
 3c73b04:      	xorl	%eax, %eax
 3c73b06:      	lock
 3c73b07:      	cmpxchgl	%ecx, (%rbx)
 3c73b0a:      	jne	0x3c7420f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x74f>
 3c73b10:      	movq	0x14cd551(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c73b17:      	movq	(%rax), %rax
 3c73b1a:      	shlq	%rax
 3c73b1d:      	testq	%rax, %rax
 3c73b20:      	jne	0x3c7421d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x75d>
 3c73b26:      	movl	$0x0, 0x20(%rsp)
 3c73b2e:      	movzbl	0x4(%rbx), %eax
 3c73b32:      	testb	%al, %al
 3c73b34:      	jne	0x3c74235 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x775>
 3c73b3a:      	cmpq	$0x0, 0x20(%rbx)
 3c73b3f:      	je	0x3c73db8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2f8>
 3c73b45:      	leaq	0x28(%rbx), %rdi
 3c73b49:      	movq	%r12, 0x10(%rsp)
 3c73b4e:      	movq	%r12, %rsi
 3c73b51:      	movq	%r15, 0x8(%rsp)
 3c73b56:      	movq	%r15, %rdx
 3c73b59:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73b5e:      	movq	%rax, %rcx
 3c73b61:      	shrq	$0x39, %rcx
 3c73b65:      	movq	0x8(%rbx), %r15
 3c73b69:      	movq	0x10(%rbx), %r12
 3c73b6d:      	movd	%ecx, %xmm0
 3c73b71:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73b75:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73b7a:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73b7f:      	xorl	%edx, %edx
 3c73b81:      	pcmpeqd	%xmm2, %xmm2
 3c73b85:      	andq	%r12, %rax
 3c73b88:      	movdqu	(%r15,%rax), %xmm3
 3c73b8e:      	movdqa	%xmm3, %xmm0
 3c73b92:      	pcmpeqb	%xmm1, %xmm0
 3c73b96:      	pmovmskb	%xmm0, %r14d
 3c73b9b:      	testl	%r14d, %r14d
 3c73b9e:      	je	0x3c73c36 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x176>
 3c73ba4:      	movq	%rax, 0x40(%rsp)
 3c73ba9:      	movq	%rbp, 0x30(%rsp)
 3c73bae:      	movdqa	%xmm1, 0xf0(%rsp)
 3c73bb7:      	movq	%rdx, 0xe0(%rsp)
 3c73bbf:      	movdqa	%xmm3, 0xd0(%rsp)
 3c73bc8:      	tzcntl	%r14d, %ecx
 3c73bcd:      	addq	%rax, %rcx
 3c73bd0:      	andq	%r12, %rcx
 3c73bd3:      	shlq	$0x8, %rcx
 3c73bd7:      	movq	%r15, %rbp
 3c73bda:      	subq	%rcx, %rbp
 3c73bdd:      	movq	0x8(%rsp), %rdx
 3c73be2:      	cmpq	-0xf0(%rbp), %rdx
 3c73be9:      	jne	0x3c73c01 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x141>
 3c73beb:      	movq	-0xf8(%rbp), %rsi
 3c73bf2:      	movq	0x10(%rsp), %rdi
 3c73bf7:      	callq	*0x14cd1ab(%rip)        # 0x5140da8 <writev+0x5140da8>
 3c73bfd:      	testl	%eax, %eax
 3c73bff:      	je	0x3c73c56 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x196>
 3c73c01:      	leal	-0x1(%r14), %eax
 3c73c05:      	andw	%r14w, %ax
 3c73c09:      	movl	%eax, %r14d
 3c73c0c:      	movq	0x40(%rsp), %rax
 3c73c11:      	movq	0x30(%rsp), %rbp
 3c73c16:      	movdqa	0xf0(%rsp), %xmm1
 3c73c1f:      	movq	0xe0(%rsp), %rdx
 3c73c27:      	pcmpeqd	%xmm2, %xmm2
 3c73c2b:      	movdqa	0xd0(%rsp), %xmm3
 3c73c34:      	jne	0x3c73bc8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x108>
 3c73c36:      	pcmpeqb	%xmm2, %xmm3
 3c73c3a:      	pmovmskb	%xmm3, %ecx
 3c73c3e:      	testl	%ecx, %ecx
 3c73c40:      	jne	0x3c73e2f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x36f>
 3c73c46:      	addq	%rdx, %rax
 3c73c49:      	addq	$0x10, %rax
 3c73c4d:      	addq	$0x10, %rdx
 3c73c51:      	jmp	0x3c73b85 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xc5>
 3c73c56:      	movb	$0x1, %r14b
 3c73c59:      	cmpb	$0x2, -0x88(%rbp)
 3c73c60:      	jae	0x3c73dbb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73c66:      	cmpq	$-0x2, -0x80(%rbp)
 3c73c6b:      	jne	0x3c73dbb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73c71:      	movq	-0x8(%rbp), %rcx
 3c73c75:      	movq	0x30(%rsp), %rax
 3c73c7a:      	cmpq	%rcx, %rax
 3c73c7d:      	movq	%rcx, %rbp
 3c73c80:      	cmovaq	%rax, %rbp
 3c73c84:      	movq	0x38(%rbx), %rdx
 3c73c88:      	xorl	%eax, %eax
 3c73c8a:      	subq	%rcx, %rdx
 3c73c8d:      	cmovaeq	%rdx, %rax
 3c73c91:      	movb	$0x2, %r14b
 3c73c94:      	addq	%rbp, %rax
 3c73c97:      	jb	0x3c73dbb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73c9d:      	cmpq	0x40(%rbx), %rax
 3c73ca1:      	ja	0x3c73dbb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73ca7:      	movq	%rax, 0x38(%rbx)
 3c73cab:      	leaq	0x28(%rbx), %rdi
 3c73caf:      	movq	0x10(%rsp), %rsi
 3c73cb4:      	movq	0x8(%rsp), %rdx
 3c73cb9:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73cbe:      	movq	%rax, %rcx
 3c73cc1:      	shrq	$0x39, %rcx
 3c73cc5:      	movd	%ecx, %xmm0
 3c73cc9:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73ccd:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73cd2:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73cd7:      	xorl	%edx, %edx
 3c73cd9:      	pcmpeqd	%xmm2, %xmm2
 3c73cdd:      	andq	%r12, %rax
 3c73ce0:      	movdqu	(%r15,%rax), %xmm3
 3c73ce6:      	movdqa	%xmm3, %xmm0
 3c73cea:      	pcmpeqb	%xmm1, %xmm0
 3c73cee:      	pmovmskb	%xmm0, %r14d
 3c73cf3:      	testl	%r14d, %r14d
 3c73cf6:      	je	0x3c73d98 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2d8>
 3c73cfc:      	movq	%rax, 0xf0(%rsp)
 3c73d04:      	movq	%rbp, 0x40(%rsp)
 3c73d09:      	movdqa	%xmm1, 0xe0(%rsp)
 3c73d12:      	movq	%rdx, 0xd0(%rsp)
 3c73d1a:      	movdqa	%xmm3, 0x100(%rsp)
 3c73d23:      	tzcntl	%r14d, %ecx
 3c73d28:      	addq	%rax, %rcx
 3c73d2b:      	andq	%r12, %rcx
 3c73d2e:      	shlq	$0x8, %rcx
 3c73d32:      	movq	%r15, %rbp
 3c73d35:      	subq	%rcx, %rbp
 3c73d38:      	movq	0x8(%rsp), %rdx
 3c73d3d:      	cmpq	-0xf0(%rbp), %rdx
 3c73d44:      	jne	0x3c73d60 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2a0>
 3c73d46:      	movq	-0xf8(%rbp), %rsi
 3c73d4d:      	movq	0x10(%rsp), %rdi
 3c73d52:      	callq	*0x14cd050(%rip)        # 0x5140da8 <writev+0x5140da8>
 3c73d58:      	testl	%eax, %eax
 3c73d5a:      	je	0x3c73e3b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x37b>
 3c73d60:      	leal	-0x1(%r14), %eax
 3c73d64:      	andw	%r14w, %ax
 3c73d68:      	movl	%eax, %r14d
 3c73d6b:      	movq	0xf0(%rsp), %rax
 3c73d73:      	movq	0x40(%rsp), %rbp
 3c73d78:      	movdqa	0xe0(%rsp), %xmm1
 3c73d81:      	movq	0xd0(%rsp), %rdx
 3c73d89:      	pcmpeqd	%xmm2, %xmm2
 3c73d8d:      	movdqa	0x100(%rsp), %xmm3
 3c73d96:      	jne	0x3c73d23 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x263>
 3c73d98:      	pcmpeqb	%xmm2, %xmm3
 3c73d9c:      	pmovmskb	%xmm3, %ecx
 3c73da0:      	testl	%ecx, %ecx
 3c73da2:      	jne	0x3c7412b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x66b>
 3c73da8:      	addq	%rdx, %rax
 3c73dab:      	addq	$0x10, %rax
 3c73daf:      	addq	$0x10, %rdx
 3c73db3:      	jmp	0x3c73cdd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x21d>
 3c73db8:      	xorl	%r14d, %r14d
 3c73dbb:      	cmpb	$0x0, 0x20(%rsp)
 3c73dc0:      	jne	0x3c73ddf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c73dc2:      	movq	0x14cd29f(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c73dc9:      	movq	(%rax), %rax
 3c73dcc:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c73dd6:      	testq	%rcx, %rax
 3c73dd9:      	jne	0x3c74277 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7b7>
 3c73ddf:      	xorl	%eax, %eax
 3c73de1:      	xchgl	%eax, (%rbx)
 3c73de3:      	cmpl	$0x2, %eax
 3c73de6:      	je	0x3c74269 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7a9>
 3c73dec:      	cmpq	$-0x1, 0x18(%r13)
 3c73df1:      	je	0x3c73dfc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x33c>
 3c73df3:      	leaq	0x18(%r13), %rdi
 3c73df7:      	callq	0x3c08ae0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c73dfc:      	movq	(%r13), %rsi
 3c73e00:      	cmpq	$-0x1, %rsi
 3c73e04:      	je	0x3c73e1a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c73e06:      	testq	%rsi, %rsi
 3c73e09:      	je	0x3c73e1a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c73e0b:      	movq	0x8(%r13), %rdi
 3c73e0f:      	movl	$0x1, %edx
 3c73e14:      	callq	*0x14ccf26(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c73e1a:      	movl	%r14d, %eax
 3c73e1d:      	addq	$0x128, %rsp            # imm = 0x128
 3c73e24:      	popq	%rbx
 3c73e25:      	popq	%r12
 3c73e27:      	popq	%r13
 3c73e29:      	popq	%r14
 3c73e2b:      	popq	%r15
 3c73e2d:      	popq	%rbp
 3c73e2e:      	retq
 3c73e2f:      	xorl	%r14d, %r14d
 3c73e32:      	cmpb	$0x0, 0x20(%rsp)
 3c73e37:      	je	0x3c73dc2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x302>
 3c73e39:      	jmp	0x3c73ddf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c73e3b:      	movq	0x40(%rsp), %rax
 3c73e40:      	movq	%rax, -0x8(%rbp)
 3c73e44:      	leaq	-0x80(%rbp), %r14
 3c73e48:      	cmpq	$-0x2, -0x80(%rbp)
 3c73e4d:      	je	0x3c73e57 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x397>
 3c73e4f:      	movq	%r14, %rdi
 3c73e52:      	callq	0x3c08970 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c73e57:      	movq	0x60(%r13), %rax
 3c73e5b:      	movq	%rax, 0x60(%r14)
 3c73e5f:      	movups	0x50(%r13), %xmm0
 3c73e64:      	movups	%xmm0, 0x50(%r14)
 3c73e69:      	movups	0x40(%r13), %xmm0
 3c73e6e:      	movups	%xmm0, 0x40(%r14)
 3c73e73:      	movdqu	(%r13), %xmm0
 3c73e79:      	movdqu	0x10(%r13), %xmm1
 3c73e7f:      	movdqu	0x20(%r13), %xmm2
 3c73e85:      	movdqu	0x30(%r13), %xmm3
 3c73e8b:      	movdqu	%xmm3, 0x30(%r14)
 3c73e91:      	movdqu	%xmm2, 0x20(%r14)
 3c73e97:      	movdqu	%xmm1, 0x10(%r14)
 3c73e9d:      	movdqu	%xmm0, (%r14)
 3c73ea2:      	cmpb	$0x0, 0x20(%rsp)
 3c73ea7:      	movq	0x14cd1ba(%rip), %r15   # 0x5141068 <writev+0x5141068>
 3c73eae:      	movq	0x8(%rsp), %r12
 3c73eb3:      	jne	0x3c73ecb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x40b>
 3c73eb5:      	movq	(%r15), %rax
 3c73eb8:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c73ec2:      	testq	%rcx, %rax
 3c73ec5:      	jne	0x3c742fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x83b>
 3c73ecb:      	xorl	%eax, %eax
 3c73ecd:      	xchgl	%eax, (%rbx)
 3c73ecf:      	cmpl	$0x2, %eax
 3c73ed2:      	je	0x3c7428e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7ce>
 3c73ed8:      	leaq	0x110(%rsp), %rdi
 3c73ee0:      	movq	0x38(%rsp), %rsi
 3c73ee5:      	movq	0x18(%rsp), %rax
 3c73eea:      	callq	*0x20(%rax)
 3c73eed:      	movq	0x120(%rsp), %rax
 3c73ef5:      	movdqa	0x110(%rsp), %xmm0
 3c73efe:      	movdqa	%xmm0, 0x50(%rsp)
 3c73f04:      	movq	%rax, 0x60(%rsp)
 3c73f09:      	movl	$0x1, %ecx
 3c73f0e:      	xorl	%eax, %eax
 3c73f10:      	lock
 3c73f11:      	cmpxchgl	%ecx, (%rbx)
 3c73f14:      	jne	0x3c74428 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x968>
 3c73f1a:      	movq	(%r15), %rax
 3c73f1d:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c73f27:      	testq	%rcx, %rax
 3c73f2a:      	jne	0x3c7429c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7dc>
 3c73f30:      	xorl	%ecx, %ecx
 3c73f32:      	movzbl	0x4(%rbx), %eax
 3c73f36:      	testb	%al, %al
 3c73f38:      	jne	0x3c742b3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7f3>
 3c73f3e:      	leaq	0x13b8b2b(%rip), %r15   # 0x502ca70 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2b0>
 3c73f45:      	movl	$0x22, %esi
 3c73f4a:      	leaq	-0x2e552a4(%rip), %rdi  # 0xe1ecad <anon.b731ebf5b89b152c0d2ad556fb42dc5b.296.llvm.12140772379724168186+0x1b5>
 3c73f51:      	cmpq	$0x0, 0x20(%rbx)
 3c73f56:      	movl	%ecx, 0x18(%rsp)
 3c73f5a:      	je	0x3c74149 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x689>
 3c73f60:      	leaq	0x28(%rbx), %rdi
 3c73f64:      	movq	0x10(%rsp), %rsi
 3c73f69:      	movq	%r12, %rdx
 3c73f6c:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73f71:      	movq	%rax, %r14
 3c73f74:      	shrq	$0x39, %rax
 3c73f78:      	movq	0x8(%rbx), %rbp
 3c73f7c:      	movq	0x10(%rbx), %rcx
 3c73f80:      	movd	%eax, %xmm0
 3c73f84:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73f88:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73f8d:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73f92:      	xorl	%edx, %edx
 3c73f94:      	pcmpeqd	%xmm2, %xmm2
 3c73f98:      	andq	%rcx, %r14
 3c73f9b:      	movdqu	(%rbp,%r14), %xmm3
 3c73fa2:      	movdqa	%xmm3, %xmm0
 3c73fa6:      	pcmpeqb	%xmm1, %xmm0
 3c73faa:      	pmovmskb	%xmm0, %r13d
 3c73faf:      	testl	%r13d, %r13d
 3c73fb2:      	je	0x3c7403b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x57b>
 3c73fb8:      	movq	%rcx, 0x38(%rsp)
 3c73fbd:      	movdqa	%xmm1, 0x20(%rsp)
 3c73fc3:      	movq	%rdx, 0x8(%rsp)
 3c73fc8:      	movdqa	%xmm3, 0x40(%rsp)
 3c73fce:      	tzcntl	%r13d, %eax
 3c73fd3:      	addq	%r14, %rax
 3c73fd6:      	andq	%rcx, %rax
 3c73fd9:      	shlq	$0x8, %rax
 3c73fdd:      	movq	%r12, %r15
 3c73fe0:      	movq	%rbp, %r12
 3c73fe3:      	subq	%rax, %r12
 3c73fe6:      	cmpq	-0xf0(%r12), %r15
 3c73fee:      	jne	0x3c7400a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x54a>
 3c73ff0:      	movq	-0xf8(%r12), %rsi
 3c73ff8:      	movq	0x10(%rsp), %rdi
 3c73ffd:      	movq	%r15, %rdx
 3c74000:      	callq	*0x14ccda2(%rip)        # 0x5140da8 <writev+0x5140da8>
 3c74006:      	testl	%eax, %eax
 3c74008:      	je	0x3c7405b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x59b>
 3c7400a:      	leal	-0x1(%r13), %eax
 3c7400e:      	andw	%r13w, %ax
 3c74012:      	movl	%eax, %r13d
 3c74015:      	movq	%r15, %r12
 3c74018:      	leaq	0x13b8a51(%rip), %r15   # 0x502ca70 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2b0>
 3c7401f:      	movq	0x38(%rsp), %rcx
 3c74024:      	movdqa	0x20(%rsp), %xmm1
 3c7402a:      	movq	0x8(%rsp), %rdx
 3c7402f:      	pcmpeqd	%xmm2, %xmm2
 3c74033:      	movdqa	0x40(%rsp), %xmm3
 3c74039:      	jne	0x3c73fce <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x50e>
 3c7403b:      	pcmpeqb	%xmm2, %xmm3
 3c7403f:      	pmovmskb	%xmm3, %eax
 3c74043:      	testl	%eax, %eax
 3c74045:      	jne	0x3c7413d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x67d>
 3c7404b:      	addq	%rdx, %r14
 3c7404e:      	addq	$0x10, %r14
 3c74052:      	addq	$0x10, %rdx
 3c74056:      	jmp	0x3c73f98 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x4d8>
 3c7405b:      	movq	-0x80(%r12), %rax
 3c74060:      	movq	$-0x2, -0x80(%r12)
 3c74069:      	cmpq	$-0x2, %rax
 3c7406d:      	je	0x3c742e3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x823>
 3c74073:      	movq	%rax, 0x68(%rsp)
 3c74078:      	movups	-0x78(%r12), %xmm0
 3c7407e:      	movdqu	-0x68(%r12), %xmm1
 3c74085:      	movdqu	-0x58(%r12), %xmm2
 3c7408c:      	movdqu	-0x48(%r12), %xmm3
 3c74093:      	movups	%xmm0, 0x70(%rsp)
 3c74098:      	movdqu	%xmm1, 0x80(%rsp)
 3c740a1:      	movdqu	%xmm2, 0x90(%rsp)
 3c740aa:      	movdqu	%xmm3, 0xa0(%rsp)
 3c740b3:      	movups	-0x38(%r12), %xmm0
 3c740b9:      	movups	%xmm0, 0xb0(%rsp)
 3c740c1:      	movdqu	-0x28(%r12), %xmm0
 3c740c8:      	movdqu	%xmm0, 0xc0(%rsp)
 3c740d1:      	cmpq	$-0x1, 0x50(%rsp)
 3c740d7:      	movq	0x30(%rsp), %r13
 3c740dc:      	movq	0x14ccf85(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c740e3:      	je	0x3c74157 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x697>
 3c740e5:      	leaq	0x68(%rsp), %rdi
 3c740ea:      	callq	0x3c08970 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c740ef:      	cmpb	$0x0, 0x18(%rsp)
 3c740f4:      	jne	0x3c7410c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x64c>
 3c740f6:      	movq	(%r14), %rax
 3c740f9:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c74103:      	testq	%rcx, %rax
 3c74106:      	jne	0x3c74320 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x860>
 3c7410c:      	xorl	%eax, %eax
 3c7410e:      	xchgl	%eax, (%rbx)
 3c74110:      	cmpl	$0x2, %eax
 3c74113:      	je	0x3c74312 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x852>
 3c74119:      	leaq	0x50(%rsp), %rdi
 3c7411e:      	callq	0x3bfff60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultIBC_uNtNtCscdodAO9FK5_5alloc6string6StringEINtNtB16_5boxed3BoxDNtNtB4_3any3AnyNtNtB4_6marker4SendEL_EEECs3pwlnhBXFtN_12memra_server>
 3c74123:      	movb	$0x4, %r14b
 3c74126:      	jmp	0x3c73e1a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c7412b:      	leaq	0x13b890e(%rip), %rdi   # 0x502ca40 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x280>
 3c74132:      	callq	*0x14ccd08(%rip)        # 0x5140e40 <writev+0x5140e40>
 3c74138:      	jmp	0x3c742e1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x821>
 3c7413d:      	movl	$0x22, %esi
 3c74142:      	leaq	-0x2e5549c(%rip), %rdi  # 0xe1ecad <anon.b731ebf5b89b152c0d2ad556fb42dc5b.296.llvm.12140772379724168186+0x1b5>
 3c74149:      	movq	%r15, %rdx
 3c7414c:      	callq	*0x14ccca6(%rip)        # 0x5140df8 <writev+0x5140df8>
 3c74152:      	jmp	0x3c742e1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x821>
 3c74157:      	leaq	-0xe8(%r12), %r14
 3c7415f:      	movq	-0x8(%r12), %r15
 3c74164:      	movq	%r14, %rdi
 3c74167:      	callq	0x3c08970 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c7416c:      	movq	0xc8(%rsp), %rax
 3c74174:      	movq	%rax, 0x60(%r14)
 3c74178:      	movups	0xb8(%rsp), %xmm0
 3c74180:      	movups	%xmm0, 0x50(%r14)
 3c74185:      	movups	0xa8(%rsp), %xmm0
 3c7418d:      	movups	%xmm0, 0x40(%r14)
 3c74192:      	movdqu	0x68(%rsp), %xmm0
 3c74198:      	movdqu	0x78(%rsp), %xmm1
 3c7419e:      	movdqu	0x88(%rsp), %xmm2
 3c741a7:      	movdqu	0x98(%rsp), %xmm3
 3c741b0:      	movdqu	%xmm3, 0x30(%r14)
 3c741b6:      	movdqu	%xmm2, 0x20(%r14)
 3c741bc:      	movdqu	%xmm1, 0x10(%r14)
 3c741c2:      	movdqu	%xmm0, (%r14)
 3c741c7:      	callq	*0x14ccce3(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3c741cd:      	movq	%rax, -0x18(%r12)
 3c741d2:      	movl	%edx, -0x10(%r12)
 3c741d7:      	movq	%r13, -0x8(%r12)
 3c741dc:      	movq	0x38(%rbx), %rax
 3c741e0:      	xorl	%ecx, %ecx
 3c741e2:      	subq	%r15, %rax
 3c741e5:      	cmovaeq	%rax, %rcx
 3c741e9:      	addq	%r13, %rcx
 3c741ec:      	movq	%rcx, 0x38(%rbx)
 3c741f0:      	movzbl	0x18(%rsp), %esi
 3c741f5:      	movq	%rbx, %rdi
 3c741f8:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c741fd:      	leaq	0x50(%rsp), %rdi
 3c74202:      	callq	0x3bfff60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultIBC_uNtNtCscdodAO9FK5_5alloc6string6StringEINtNtB16_5boxed3BoxDNtNtB4_3any3AnyNtNtB4_6marker4SendEL_EEECs3pwlnhBXFtN_12memra_server>
 3c74207:      	movb	$-0x1, %r14b
 3c7420a:      	jmp	0x3c73e1a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c7420f:      	movq	%rbx, %rdi
 3c74212:      	callq	*0x14cd1f0(%rip)        # 0x5141408 <writev+0x5141408>
 3c74218:      	jmp	0x3c73b10 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x50>
 3c7421d:      	callq	*0x14cce55(%rip)        # 0x5141078 <writev+0x5141078>
 3c74223:      	xorb	$0x1, %al
 3c74225:      	movl	%eax, 0x20(%rsp)
 3c74229:      	movzbl	0x4(%rbx), %eax
 3c7422d:      	testb	%al, %al
 3c7422f:      	je	0x3c73b3a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7a>
 3c74235:      	movq	%rbx, 0x68(%rsp)
 3c7423a:      	movl	0x20(%rsp), %eax
 3c7423e:      	movb	%al, 0x70(%rsp)
 3c74242:      	leaq	-0x2e55868(%rip), %rdi  # 0xe1e9e1 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c74249:      	leaq	0x13b82a0(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c74250:      	leaq	0x13b87d1(%rip), %r8    # 0x502ca28 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x268>
 3c74257:      	leaq	0x68(%rsp), %rdx
 3c7425c:      	movl	$0x2b, %esi
 3c74261:      	callq	*0x14ccc31(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c74267:      	jmp	0x3c742e1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x821>
 3c74269:      	movq	%rbx, %rdi
 3c7426c:      	callq	*0x14ccdfe(%rip)        # 0x5141070 <writev+0x5141070>
 3c74272:      	jmp	0x3c73dec <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x32c>
 3c74277:      	callq	*0x14ccdfb(%rip)        # 0x5141078 <writev+0x5141078>
 3c7427d:      	testb	%al, %al
 3c7427f:      	jne	0x3c73ddf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c74285:      	movb	$0x1, 0x4(%rbx)
 3c74289:      	jmp	0x3c73ddf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c7428e:      	movq	%rbx, %rdi
 3c74291:      	callq	*0x14ccdd9(%rip)        # 0x5141070 <writev+0x5141070>
 3c74297:      	jmp	0x3c73ed8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x418>
 3c7429c:      	callq	*0x14ccdd6(%rip)        # 0x5141078 <writev+0x5141078>
 3c742a2:      	movl	%eax, %ecx
 3c742a4:      	xorb	$0x1, %cl
 3c742a7:      	movzbl	0x4(%rbx), %eax
 3c742ab:      	testb	%al, %al
 3c742ad:      	je	0x3c73f3e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x47e>
 3c742b3:      	movq	%rbx, 0x68(%rsp)
 3c742b8:      	movb	%cl, 0x70(%rsp)
 3c742bc:      	leaq	-0x2e558e2(%rip), %rdi  # 0xe1e9e1 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c742c3:      	leaq	0x13b8226(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c742ca:      	leaq	0x13b8787(%rip), %r8    # 0x502ca58 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x298>
 3c742d1:      	leaq	0x68(%rsp), %rdx
 3c742d6:      	movl	$0x2b, %esi
 3c742db:      	callq	*0x14ccbb7(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c742e1:      	ud2
 3c742e3:      	leaq	0x13b879e(%rip), %r15   # 0x502ca88 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2c8>
 3c742ea:      	movl	$0x1a, %esi
 3c742ef:      	leaq	-0x2e55627(%rip), %rdi  # 0xe1eccf <anon.b731ebf5b89b152c0d2ad556fb42dc5b.296.llvm.12140772379724168186+0x1d7>
 3c742f6:      	jmp	0x3c74149 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x689>
 3c742fb:      	callq	*0x14ccd77(%rip)        # 0x5141078 <writev+0x5141078>
 3c74301:      	testb	%al, %al
 3c74303:      	jne	0x3c73ecb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x40b>
 3c74309:      	movb	$0x1, 0x4(%rbx)
 3c7430d:      	jmp	0x3c73ecb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x40b>
 3c74312:      	movq	%rbx, %rdi
 3c74315:      	callq	*0x14ccd55(%rip)        # 0x5141070 <writev+0x5141070>
 3c7431b:      	jmp	0x3c74119 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x659>
 3c74320:      	callq	*0x14ccd52(%rip)        # 0x5141078 <writev+0x5141078>
 3c74326:      	testb	%al, %al
 3c74328:      	jne	0x3c7410c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x64c>
 3c7432e:      	movb	$0x1, 0x4(%rbx)
 3c74332:      	jmp	0x3c7410c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x64c>
 3c74337:      	movq	%rax, %r15
 3c7433a:      	movq	0xc8(%rsp), %rax
 3c74342:      	movq	%rax, 0x60(%r14)
 3c74346:      	movups	0xb8(%rsp), %xmm0
 3c7434e:      	movups	%xmm0, 0x50(%r14)
 3c74353:      	movups	0xa8(%rsp), %xmm0
 3c7435b:      	movups	%xmm0, 0x40(%r14)
 3c74360:      	movdqu	0x68(%rsp), %xmm0
 3c74366:      	movdqu	0x78(%rsp), %xmm1
 3c7436c:      	movdqu	0x88(%rsp), %xmm2
 3c74375:      	movdqu	0x98(%rsp), %xmm3
 3c7437e:      	movdqu	%xmm3, 0x30(%r14)
 3c74384:      	movdqu	%xmm2, 0x20(%r14)
 3c7438a:      	movdqu	%xmm1, 0x10(%r14)
 3c74390:      	movdqu	%xmm0, (%r14)
 3c74395:      	jmp	0x3c74499 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9d9>
 3c7439a:      	movq	%rax, %r15
 3c7439d:      	movq	0x60(%r13), %rax
 3c743a1:      	movq	%rax, 0x60(%r14)
 3c743a5:      	movups	0x50(%r13), %xmm0
 3c743aa:      	movups	%xmm0, 0x50(%r14)
 3c743af:      	movups	0x40(%r13), %xmm0
 3c743b4:      	movups	%xmm0, 0x40(%r14)
 3c743b9:      	movdqu	(%r13), %xmm0
 3c743bf:      	movdqu	0x10(%r13), %xmm1
 3c743c5:      	movdqu	0x20(%r13), %xmm2
 3c743cb:      	movdqu	0x30(%r13), %xmm3
 3c743d1:      	movdqu	%xmm3, 0x30(%r14)
 3c743d7:      	movdqu	%xmm2, 0x20(%r14)
 3c743dd:      	movdqu	%xmm1, 0x10(%r14)
 3c743e3:      	movdqu	%xmm0, (%r14)
 3c743e8:      	xorl	%ebp, %ebp
 3c743ea:      	jmp	0x3c744b8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9f8>
 3c743ef:      	movq	%rax, %rdi
 3c743f2:      	callq	*0x14cca10(%rip)        # 0x5140e08 <writev+0x5140e08>
 3c743f8:      	movq	%rax, 0x58(%rsp)
 3c743fd:      	movq	%rdx, 0x60(%rsp)
 3c74402:      	movq	$-0x2, 0x50(%rsp)
 3c7440b:      	movq	0x14ccc56(%rip), %r15   # 0x5141068 <writev+0x5141068>
 3c74412:      	movq	0x8(%rsp), %r12
 3c74417:      	movl	$0x1, %ecx
 3c7441c:      	xorl	%eax, %eax
 3c7441e:      	lock
 3c7441f:      	cmpxchgl	%ecx, (%rbx)
 3c74422:      	je	0x3c73f1a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x45a>
 3c74428:      	movq	%rbx, %rdi
 3c7442b:      	callq	*0x14ccfd7(%rip)        # 0x5141408 <writev+0x5141408>
 3c74431:      	jmp	0x3c73f1a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x45a>
 3c74436:      	callq	*0x14cc9e4(%rip)        # 0x5140e20 <writev+0x5140e20>
 3c7443c:      	movq	%rax, %r15
 3c7443f:      	jmp	0x3c744a6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9e6>
 3c74441:      	movq	%rax, %r15
 3c74444:      	jmp	0x3c744cc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa0c>
 3c74449:      	movq	%rax, %r15
 3c7444c:      	movq	(%r13), %rsi
 3c74450:      	testq	%rsi, %rsi
 3c74453:      	jle	0x3c744d4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa14>
 3c74455:      	movq	0x8(%r13), %rdi
 3c74459:      	movl	$0x1, %edx
 3c7445e:      	callq	*0x14cc8dc(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c74464:      	movq	%r15, %rdi
 3c74467:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c7446c:      	movq	%rax, %r15
 3c7446f:      	leaq	0x68(%rsp), %rdi
 3c74474:      	callq	0x3c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c74479:      	jmp	0x3c744a6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9e6>
 3c7447b:      	callq	*0x14cc937(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c74481:      	movq	%rax, %r15
 3c74484:      	leaq	0x68(%rsp), %rdi
 3c74489:      	callq	0x3c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c7448e:      	jmp	0x3c744cc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa0c>
 3c74490:      	callq	*0x14cc922(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c74496:      	movq	%rax, %r15
 3c74499:      	movzbl	0x18(%rsp), %esi
 3c7449e:      	movq	%rbx, %rdi
 3c744a1:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c744a6:      	leaq	0x50(%rsp), %rdi
 3c744ab:      	callq	0x3bfff60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultIBC_uNtNtCscdodAO9FK5_5alloc6string6StringEINtNtB16_5boxed3BoxDNtNtB4_3any3AnyNtNtB4_6marker4SendEL_EEECs3pwlnhBXFtN_12memra_server>
 3c744b0:      	jmp	0x3c744d4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa14>
 3c744b2:      	movq	%rax, %r15
 3c744b5:      	movb	$0x1, %bpl
 3c744b8:      	movl	0x20(%rsp), %eax
 3c744bc:      	movzbl	%al, %esi
 3c744bf:      	movq	%rbx, %rdi
 3c744c2:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c744c7:      	testb	%bpl, %bpl
 3c744ca:      	je	0x3c744d4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa14>
 3c744cc:      	movq	%r13, %rdi
 3c744cf:      	callq	0x3c08970 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c744d4:      	movq	%r15, %rdi
 3c744d7:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c744dc:      	callq	*0x14cc8d6(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c744e2:      	int3
 3c744e3:      	int3
 3c744e4:      	int3
 3c744e5:      	int3
 3c744e6:      	int3
 3c744e7:      	int3
 3c744e8:      	int3
 3c744e9:      	int3
 3c744ea:      	int3
 3c744eb:      	int3
 3c744ec:      	int3
 3c744ed:      	int3
 3c744ee:      	int3
 3c744ef:      	int3

0000000003c744f0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get>:
 3c744f0:      	pushq	%rbp
 3c744f1:      	pushq	%r15
 3c744f3:      	pushq	%r14
 3c744f5:      	pushq	%r13
 3c744f7:      	pushq	%r12
 3c744f9:      	pushq	%rbx
 3c744fa:      	subq	$0x108, %rsp            # imm = 0x108
 3c74501:      	movq	%rcx, %r13
 3c74504:      	movq	%rdx, %r12
 3c74507:      	movq	%rsi, %rbx
 3c7450a:      	movq	%rdi, %r14
 3c7450d:      	movl	$0x1, %ecx
 3c74512:      	xorl	%eax, %eax
 3c74514:      	lock
 3c74515:      	cmpxchgl	%ecx, (%rsi)
 3c74518:      	jne	0x3c74909 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x419>
 3c7451e:      	movq	0x14ccb43(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c74525:      	movq	(%rax), %rax
 3c74528:      	shlq	%rax
 3c7452b:      	testq	%rax, %rax
 3c7452e:      	jne	0x3c74917 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x427>
 3c74534:      	xorl	%r15d, %r15d
 3c74537:      	movzbl	0x4(%rbx), %eax
 3c7453b:      	testb	%al, %al
 3c7453d:      	jne	0x3c74930 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x440>
 3c74543:      	leaq	0x8(%rbx), %rsi
 3c74547:      	movq	%rbx, %rdi
 3c7454a:      	callq	0x3c47600 <_RNvMNtCs3pwlnhBXFtN_12memra_server9job_storeNtB2_16InMemoryJobStore12sweep_locked.llvm.12140772379724168186>
 3c7454f:      	cmpq	$0x0, 0x20(%rbx)
 3c74554:      	je	0x3c74691 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x1a1>
 3c7455a:      	movl	%r15d, 0xc(%rsp)
 3c7455f:      	leaq	0x28(%rbx), %rdi
 3c74563:      	movq	%r12, %rsi
 3c74566:      	movq	%r13, %rdx
 3c74569:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c7456e:      	movq	%rax, %rbp
 3c74571:      	shrq	$0x39, %rax
 3c74575:      	movq	0x8(%rbx), %rcx
 3c74579:      	movq	0x10(%rbx), %rdx
 3c7457d:      	movd	%eax, %xmm0
 3c74581:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74585:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c7458a:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c7458f:      	xorl	%esi, %esi
 3c74591:      	pcmpeqd	%xmm2, %xmm2
 3c74595:      	movq	0x14cc80c(%rip), %r8    # 0x5140da8 <writev+0x5140da8>
 3c7459c:      	andq	%rdx, %rbp
 3c7459f:      	movdqu	(%rcx,%rbp), %xmm3
 3c745a4:      	movdqa	%xmm3, %xmm0
 3c745a8:      	pcmpeqb	%xmm1, %xmm0
 3c745ac:      	pmovmskb	%xmm0, %r15d
 3c745b1:      	testl	%r15d, %r15d
 3c745b4:      	je	0x3c74670 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x180>
 3c745ba:      	movq	%r12, 0x18(%rsp)
 3c745bf:      	movq	%rcx, 0x78(%rsp)
 3c745c4:      	movq	%rdx, 0x70(%rsp)
 3c745c9:      	movdqa	%xmm1, 0xe0(%rsp)
 3c745d2:      	movq	%rsi, 0x68(%rsp)
 3c745d7:      	movdqa	%xmm3, 0xd0(%rsp)
 3c745e0:      	tzcntl	%r15d, %eax
 3c745e5:      	addq	%rbp, %rax
 3c745e8:      	andq	%rdx, %rax
 3c745eb:      	shlq	$0x8, %rax
 3c745ef:      	movq	%r13, %r12
 3c745f2:      	movq	%rcx, %r13
 3c745f5:      	subq	%rax, %r13
 3c745f8:      	cmpq	-0xf0(%r13), %r12
 3c745ff:      	jne	0x3c7462d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x13d>
 3c74601:      	movq	-0xf8(%r13), %rsi
 3c74608:      	movq	0x18(%rsp), %rdi
 3c7460d:      	movq	%r12, %rdx
 3c74610:      	movq	%rbx, 0x10(%rsp)
 3c74615:      	movq	%r14, %rbx
 3c74618:      	movq	%r8, %r14
 3c7461b:      	callq	*%r8
 3c7461e:      	movq	%r14, %r8
 3c74621:      	movq	%rbx, %r14
 3c74624:      	movq	0x10(%rsp), %rbx
 3c74629:      	testl	%eax, %eax
 3c7462b:      	je	0x3c746a6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x1b6>
 3c7462d:      	leal	-0x1(%r15), %eax
 3c74631:      	andw	%r15w, %ax
 3c74635:      	movl	%eax, %r15d
 3c74638:      	movq	%r12, %r13
 3c7463b:      	movq	0x18(%rsp), %r12
 3c74640:      	movq	0x78(%rsp), %rcx
 3c74645:      	movq	0x70(%rsp), %rdx
 3c7464a:      	movdqa	0xe0(%rsp), %xmm1
 3c74653:      	movq	0x68(%rsp), %rsi
 3c74658:      	pcmpeqd	%xmm2, %xmm2
 3c7465c:      	movdqa	0xd0(%rsp), %xmm3
 3c74665:      	jne	0x3c745e0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0xf0>
 3c7466b:      	nopl	(%rax,%rax)
 3c74670:      	pcmpeqb	%xmm2, %xmm3
 3c74674:      	pmovmskb	%xmm3, %eax
 3c74678:      	testl	%eax, %eax
 3c7467a:      	movl	0xc(%rsp), %r15d
 3c7467f:      	jne	0x3c74691 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x1a1>
 3c74681:      	addq	%rsi, %rbp
 3c74684:      	addq	$0x10, %rbp
 3c74688:      	addq	$0x10, %rsi
 3c7468c:      	jmp	0x3c7459c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0xac>
 3c74691:      	movq	$-0x2, (%r14)
 3c74698:      	testb	%r15b, %r15b
 3c7469b:      	je	0x3c748ce <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3de>
 3c746a1:      	jmp	0x3c748eb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c746a6:      	movzbl	-0x88(%r13), %ebp
 3c746ae:      	movq	-0xd0(%r13), %rcx
 3c746b5:      	cmpq	$-0x1, %rcx
 3c746b9:      	je	0x3c74700 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x210>
 3c746bb:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c746c5:      	incq	%rax
 3c746c8:      	movq	%rcx, %rdx
 3c746cb:      	xorq	%rax, %rdx
 3c746ce:      	testq	%rcx, %rcx
 3c746d1:      	movl	$0x5, %ecx
 3c746d6:      	cmovsq	%rdx, %rcx
 3c746da:      	leaq	-0xd0(%r13), %rsi
 3c746e1:      	leaq	-0x2e5623c(%rip), %rdx  # 0xe1e4ac <anon.579209908c49646e156899b92400b4a2.287.llvm.9148498120539102879+0x148d>
 3c746e8:      	movslq	(%rdx,%rcx,4), %rcx
 3c746ec:      	addq	%rdx, %rcx
 3c746ef:      	movl	0xc(%rsp), %r15d
 3c746f4:      	jmpq	*%rcx
 3c746f6:      	movq	%rax, 0x20(%rsp)
 3c746fb:      	jmp	0x3c747ed <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c74700:      	movq	$-0x1, 0x80(%rsp)
 3c7470c:      	movl	0xc(%rsp), %r15d
 3c74711:      	addq	$-0xe8, %r13
 3c74718:      	cmpq	$-0x1, (%r13)
 3c7471d:      	jne	0x3c74853 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x363>
 3c74723:      	movq	$-0x1, %rax
 3c7472a:      	jmp	0x3c74875 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x385>
 3c7472f:      	leaq	0x28(%rsp), %rdi
 3c74734:      	movq	-0xc0(%r13), %rsi
 3c7473b:      	movq	-0xb8(%r13), %rdx
 3c74742:      	callq	0x3afa8e0 <_RINvXNvMNtCscdodAO9FK5_5alloc5sliceSp9to_vec_inNtNtCs8OSp0AlFmbY_10serde_json5value5ValueNtB3_10ConvertVec6to_vecNtNtB8_5alloc6GlobalECs3pwlnhBXFtN_12memra_server.llvm.5605875074211559972>
 3c74747:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c74751:      	addq	$0x5, %rax
 3c74755:      	movq	%rax, 0x20(%rsp)
 3c7475a:      	jmp	0x3c747ed <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c7475f:      	leaq	-0xc8(%r13), %rsi
 3c74766:      	leaq	0x28(%rsp), %rdi
 3c7476b:      	callq	*0x14cc827(%rip)        # 0x5140f98 <writev+0x5140f98>
 3c74771:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c7477b:      	addq	$0x3, %rax
 3c7477f:      	movq	%rax, 0x20(%rsp)
 3c74784:      	jmp	0x3c747ed <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c74786:      	leaq	-0xc8(%r13), %rsi
 3c7478d:      	leaq	0x28(%rsp), %rdi
 3c74792:      	callq	*0x14cc800(%rip)        # 0x5140f98 <writev+0x5140f98>
 3c74798:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c747a2:      	addq	$0x4, %rax
 3c747a6:      	movq	%rax, 0x20(%rsp)
 3c747ab:      	jmp	0x3c747ed <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c747ad:      	movq	0x40(%rsi), %rax
 3c747b1:      	movq	%rax, 0x60(%rsp)
 3c747b6:      	movdqu	(%rsi), %xmm0
 3c747ba:      	movdqu	0x10(%rsi), %xmm1
 3c747bf:      	movdqu	0x20(%rsi), %xmm2
 3c747c4:      	movdqu	0x30(%rsi), %xmm3
 3c747c9:      	movdqa	%xmm3, 0x50(%rsp)
 3c747cf:      	movdqa	%xmm2, 0x40(%rsp)
 3c747d5:      	movdqa	%xmm1, 0x30(%rsp)
 3c747db:      	movdqa	%xmm0, 0x20(%rsp)
 3c747e1:      	jmp	0x3c747ed <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c747e3:      	leaq	0x20(%rsp), %rdi
 3c747e8:      	callq	0x3a01470 <_RNvXNtCseAOs6zoTI9e_8indexmap3mapINtB2_8IndexMapNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtCs4NRVxsYgnAr_4core5clone5Clone5cloneCs3pwlnhBXFtN_12memra_server>
 3c747ed:      	movq	0x60(%rsp), %rax
 3c747f2:      	movq	%rax, 0xc0(%rsp)
 3c747fa:      	movq	0x20(%rsp), %rax
 3c747ff:      	movq	0x28(%rsp), %rcx
 3c74804:      	movdqa	0x30(%rsp), %xmm0
 3c7480a:      	movdqa	0x40(%rsp), %xmm1
 3c74810:      	movdqa	0x50(%rsp), %xmm2
 3c74816:      	movdqa	%xmm2, 0xb0(%rsp)
 3c7481f:      	movdqa	%xmm1, 0xa0(%rsp)
 3c74828:      	movdqa	%xmm0, 0x90(%rsp)
 3c74831:      	movq	%rax, 0x80(%rsp)
 3c74839:      	movq	%rcx, 0x88(%rsp)
 3c74841:      	addq	$-0xe8, %r13
 3c74848:      	cmpq	$-0x1, (%r13)
 3c7484d:      	je	0x3c74723 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x233>
 3c74853:      	leaq	0x20(%rsp), %rdi
 3c74858:      	movq	%r13, %rsi
 3c7485b:      	callq	*0x14cc737(%rip)        # 0x5140f98 <writev+0x5140f98>
 3c74861:      	movq	0x20(%rsp), %rax
 3c74866:      	movdqu	0x28(%rsp), %xmm0
 3c7486c:      	movdqa	%xmm0, 0xf0(%rsp)
 3c74875:      	movq	0xc0(%rsp), %rcx
 3c7487d:      	movq	%rcx, 0x58(%r14)
 3c74881:      	movaps	0x80(%rsp), %xmm0
 3c74889:      	movaps	0x90(%rsp), %xmm1
 3c74891:      	movaps	0xa0(%rsp), %xmm2
 3c74899:      	movaps	0xb0(%rsp), %xmm3
 3c748a1:      	movups	%xmm3, 0x48(%r14)
 3c748a6:      	movups	%xmm2, 0x38(%r14)
 3c748ab:      	movups	%xmm1, 0x28(%r14)
 3c748b0:      	movups	%xmm0, 0x18(%r14)
 3c748b5:      	movq	%rax, (%r14)
 3c748b8:      	movaps	0xf0(%rsp), %xmm0
 3c748c0:      	movups	%xmm0, 0x8(%r14)
 3c748c5:      	movb	%bpl, 0x60(%r14)
 3c748c9:      	testb	%r15b, %r15b
 3c748cc:      	jne	0x3c748eb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c748ce:      	movq	0x14cc793(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c748d5:      	movq	(%rax), %rax
 3c748d8:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c748e2:      	testq	%rcx, %rax
 3c748e5:      	jne	0x3c7496c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x47c>
 3c748eb:      	xorl	%eax, %eax
 3c748ed:      	xchgl	%eax, (%rbx)
 3c748ef:      	cmpl	$0x2, %eax
 3c748f2:      	je	0x3c74961 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x471>
 3c748f4:      	movq	%r14, %rax
 3c748f7:      	addq	$0x108, %rsp            # imm = 0x108
 3c748fe:      	popq	%rbx
 3c748ff:      	popq	%r12
 3c74901:      	popq	%r13
 3c74903:      	popq	%r14
 3c74905:      	popq	%r15
 3c74907:      	popq	%rbp
 3c74908:      	retq
 3c74909:      	movq	%rbx, %rdi
 3c7490c:      	callq	*0x14ccaf6(%rip)        # 0x5141408 <writev+0x5141408>
 3c74912:      	jmp	0x3c7451e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2e>
 3c74917:      	callq	*0x14cc75b(%rip)        # 0x5141078 <writev+0x5141078>
 3c7491d:      	movl	%eax, %r15d
 3c74920:      	xorb	$0x1, %r15b
 3c74924:      	movzbl	0x4(%rbx), %eax
 3c74928:      	testb	%al, %al
 3c7492a:      	je	0x3c74543 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x53>
 3c74930:      	movq	%rbx, 0x20(%rsp)
 3c74935:      	movb	%r15b, 0x28(%rsp)
 3c7493a:      	leaq	-0x2e55f60(%rip), %rdi  # 0xe1e9e1 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c74941:      	leaq	0x13b7ba8(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c74948:      	leaq	0x13b8151(%rip), %r8    # 0x502caa0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2e0>
 3c7494f:      	leaq	0x20(%rsp), %rdx
 3c74954:      	movl	$0x2b, %esi
 3c74959:      	callq	*0x14cc539(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c7495f:      	ud2
 3c74961:      	movq	%rbx, %rdi
 3c74964:      	callq	*0x14cc706(%rip)        # 0x5141070 <writev+0x5141070>
 3c7496a:      	jmp	0x3c748f4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x404>
 3c7496c:      	callq	*0x14cc706(%rip)        # 0x5141078 <writev+0x5141078>
 3c74972:      	testb	%al, %al
 3c74974:      	jne	0x3c748eb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c7497a:      	movb	$0x1, 0x4(%rbx)
 3c7497e:      	jmp	0x3c748eb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c74983:      	movq	%rax, %r14
 3c74986:      	cmpq	$-0x1, 0x80(%rsp)
 3c7498f:      	je	0x3c749b3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x4c3>
 3c74991:      	leaq	0x80(%rsp), %rdi
 3c74999:      	callq	0x3c08ae0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c7499e:      	jmp	0x3c749b3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x4c3>
 3c749a0:      	callq	*0x14cc412(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c749a6:      	movq	%rbx, 0x10(%rsp)
 3c749ab:      	movl	%r15d, 0xc(%rsp)
 3c749b0:      	movq	%rax, %r14
 3c749b3:      	movzbl	0xc(%rsp), %esi
 3c749b8:      	movq	0x10(%rsp), %rdi
 3c749bd:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c749c2:      	jmp	0x3c749d7 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x4e7>
 3c749c4:      	callq	*0x14cc3ee(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c749ca:      	movq	%rax, %r14
 3c749cd:      	leaq	0x20(%rsp), %rdi
 3c749d2:      	callq	0x3c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c749d7:      	movq	%r14, %rdi
 3c749da:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c749df:      	callq	*0x14cc3d3(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c749e5:      	int3
 3c749e6:      	int3
 3c749e7:      	int3
 3c749e8:      	int3
 3c749e9:      	int3
 3c749ea:      	int3
 3c749eb:      	int3
 3c749ec:      	int3
 3c749ed:      	int3
 3c749ee:      	int3
 3c749ef:      	int3

0000000003c749f0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put>:
 3c749f0:      	pushq	%rbp
 3c749f1:      	pushq	%r15
 3c749f3:      	pushq	%r14
 3c749f5:      	pushq	%r13
 3c749f7:      	pushq	%r12
 3c749f9:      	pushq	%rbx
 3c749fa:      	subq	$0x298, %rsp            # imm = 0x298
 3c74a01:      	movq	%rcx, %r13
 3c74a04:      	movq	%rdx, %r12
 3c74a07:      	movq	%rsi, 0x18(%rsp)
 3c74a0c:      	movq	%rdi, %r14
 3c74a0f:      	movl	$0x1, %ecx
 3c74a14:      	xorl	%eax, %eax
 3c74a16:      	lock
 3c74a17:      	cmpxchgl	%ecx, (%rdi)
 3c74a1a:      	jne	0x3c75103 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x713>
 3c74a20:      	movq	0x14cc641(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c74a27:      	movq	(%rax), %rax
 3c74a2a:      	shlq	%rax
 3c74a2d:      	testq	%rax, %rax
 3c74a30:      	jne	0x3c75111 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x721>
 3c74a36:      	xorl	%eax, %eax
 3c74a38:      	movzbl	0x4(%r14), %ecx
 3c74a3d:      	testb	%cl, %cl
 3c74a3f:      	jne	0x3c75126 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x736>
 3c74a45:      	movl	%eax, 0x24(%rsp)
 3c74a49:      	leaq	0x8(%r14), %rsi
 3c74a4d:      	movb	$0x1, %al
 3c74a4f:      	movl	%eax, 0x4(%rsp)
 3c74a53:      	movq	%r14, %rdi
 3c74a56:      	callq	0x3c47600 <_RNvMNtCs3pwlnhBXFtN_12memra_server9job_storeNtB2_16InMemoryJobStore12sweep_locked.llvm.12140772379724168186>
 3c74a5b:      	cmpq	$0x0, 0x20(%r14)
 3c74a60:      	movq	%r12, 0x10(%rsp)
 3c74a65:      	je	0x3c74b71 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x181>
 3c74a6b:      	leaq	0x28(%r14), %rdi
 3c74a6f:      	movq	0x18(%rsp), %rsi
 3c74a74:      	movq	%r12, %rdx
 3c74a77:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c74a7c:      	movq	%rax, %rcx
 3c74a7f:      	shrq	$0x39, %rcx
 3c74a83:      	movq	0x8(%r14), %r15
 3c74a87:      	movq	0x10(%r14), %rbp
 3c74a8b:      	movd	%ecx, %xmm0
 3c74a8f:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74a93:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c74a98:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c74a9d:      	xorl	%edx, %edx
 3c74a9f:      	pcmpeqd	%xmm2, %xmm2
 3c74aa3:      	movq	0x14cc2fe(%rip), %rbx   # 0x5140da8 <writev+0x5140da8>
 3c74aaa:      	andq	%rbp, %rax
 3c74aad:      	movdqu	(%r15,%rax), %xmm3
 3c74ab3:      	movdqa	%xmm3, %xmm0
 3c74ab7:      	pcmpeqb	%xmm1, %xmm0
 3c74abb:      	pmovmskb	%xmm0, %r12d
 3c74ac0:      	testl	%r12d, %r12d
 3c74ac3:      	je	0x3c74b50 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x160>
 3c74ac9:      	movq	%r13, 0x28(%rsp)
 3c74ace:      	movdqa	%xmm1, 0x30(%rsp)
 3c74ad4:      	movq	%rdx, 0x8(%rsp)
 3c74ad9:      	movq	%rax, 0x60(%rsp)
 3c74ade:      	movdqa	%xmm3, 0x50(%rsp)
 3c74ae4:      	tzcntl	%r12d, %ecx
 3c74ae9:      	addq	%rax, %rcx
 3c74aec:      	andq	%rbp, %rcx
 3c74aef:      	shlq	$0x8, %rcx
 3c74af3:      	movq	%r15, %r13
 3c74af6:      	subq	%rcx, %r13
 3c74af9:      	movq	0x10(%rsp), %rdx
 3c74afe:      	cmpq	-0xf0(%r13), %rdx
 3c74b05:      	jne	0x3c74b19 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x129>
 3c74b07:      	movq	-0xf8(%r13), %rsi
 3c74b0e:      	movq	0x18(%rsp), %rdi
 3c74b13:      	callq	*%rbx
 3c74b15:      	testl	%eax, %eax
 3c74b17:      	je	0x3c74b8a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x19a>
 3c74b19:      	leal	-0x1(%r12), %eax
 3c74b1e:      	andw	%r12w, %ax
 3c74b22:      	movl	%eax, %r12d
 3c74b25:      	movq	0x60(%rsp), %rax
 3c74b2a:      	movq	0x28(%rsp), %r13
 3c74b2f:      	movdqa	0x30(%rsp), %xmm1
 3c74b35:      	movq	0x8(%rsp), %rdx
 3c74b3a:      	pcmpeqd	%xmm2, %xmm2
 3c74b3e:      	movdqa	0x50(%rsp), %xmm3
 3c74b44:      	jne	0x3c74ae4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0xf4>
 3c74b46:      	nopw	%cs:(%rax,%rax)
 3c74b50:      	pcmpeqb	%xmm2, %xmm3
 3c74b54:      	pmovmskb	%xmm3, %ecx
 3c74b58:      	testl	%ecx, %ecx
 3c74b5a:      	movq	0x10(%rsp), %r12
 3c74b5f:      	jne	0x3c74b71 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x181>
 3c74b61:      	addq	%rdx, %rax
 3c74b64:      	addq	$0x10, %rax
 3c74b68:      	addq	$0x10, %rdx
 3c74b6c:      	jmp	0x3c74aaa <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0xba>
 3c74b71:      	cmpb	$0x0, 0x60(%r13)
 3c74b76:      	je	0x3c74bc0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x1d0>
 3c74b78:      	xorl	%ebx, %ebx
 3c74b7a:      	cmpb	$0x0, 0x24(%rsp)
 3c74b7f:      	je	0x3c74e99 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4a9>
 3c74b85:      	jmp	0x3c74eb6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c74b8a:      	movb	$0x1, %bl
 3c74b8c:      	cmpb	$0x1, -0x88(%r13)
 3c74b94:      	jbe	0x3c74bab <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x1bb>
 3c74b96:      	movq	0x28(%rsp), %r13
 3c74b9b:      	cmpb	$0x0, 0x24(%rsp)
 3c74ba0:      	je	0x3c74e99 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4a9>
 3c74ba6:      	jmp	0x3c74eb6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c74bab:      	cmpq	$-0x2, -0x80(%r13)
 3c74bb0:      	movq	0x28(%rsp), %r13
 3c74bb5:      	movq	0x10(%rsp), %r12
 3c74bba:      	jne	0x3c74e92 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4a2>
 3c74bc0:      	movq	%r13, %rdi
 3c74bc3:      	callq	0x3c6efb0 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store17record_size_bytes>
 3c74bc8:      	movq	%rax, %r15
 3c74bcb:      	cmpb	$0x1, 0x60(%r13)
 3c74bd0:      	movq	%r13, 0x28(%rsp)
 3c74bd5:      	jbe	0x3c74bf9 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x209>
 3c74bd7:      	callq	*0x14cc2d3(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3c74bdd:      	movq	%rax, 0x60(%rsp)
 3c74be2:      	cmpq	$0x0, 0x20(%r14)
 3c74be7:      	jne	0x3c74d40 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x350>
 3c74bed:      	movl	%edx, 0x30(%rsp)
 3c74bf1:      	xorl	%r13d, %r13d
 3c74bf4:      	jmp	0x3c74e6f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47f>
 3c74bf9:      	cmpq	$0x0, 0x20(%r14)
 3c74bfe:      	je	0x3c74d2b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x33b>
 3c74c04:      	leaq	0x28(%r14), %rdi
 3c74c08:      	movq	0x18(%rsp), %rsi
 3c74c0d:      	movq	%r12, %rdx
 3c74c10:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c74c15:      	movq	%rax, %rcx
 3c74c18:      	shrq	$0x39, %rcx
 3c74c1c:      	movq	0x8(%r14), %r13
 3c74c20:      	movq	0x10(%r14), %rdi
 3c74c24:      	movd	%ecx, %xmm0
 3c74c28:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74c2c:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c74c31:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c74c36:      	xorl	%esi, %esi
 3c74c38:      	pcmpeqd	%xmm2, %xmm2
 3c74c3c:      	movq	0x14cc165(%rip), %rbp   # 0x5140da8 <writev+0x5140da8>
 3c74c43:      	andq	%rdi, %rax
 3c74c46:      	movdqu	(%r13,%rax), %xmm3
 3c74c4d:      	movdqa	%xmm3, %xmm0
 3c74c51:      	pcmpeqb	%xmm1, %xmm0
 3c74c55:      	pmovmskb	%xmm0, %r12d
 3c74c5a:      	testl	%r12d, %r12d
 3c74c5d:      	je	0x3c74ced <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x2fd>
 3c74c63:      	movq	%rax, 0x30(%rsp)
 3c74c68:      	movq	%r15, 0x8(%rsp)
 3c74c6d:      	movdqa	%xmm1, 0x60(%rsp)
 3c74c73:      	movq	%rsi, 0x50(%rsp)
 3c74c78:      	movq	%rdi, 0x48(%rsp)
 3c74c7d:      	movdqa	%xmm3, 0x70(%rsp)
 3c74c83:      	tzcntl	%r12d, %ecx
 3c74c88:      	addq	%rax, %rcx
 3c74c8b:      	andq	%rdi, %rcx
 3c74c8e:      	shlq	$0x8, %rcx
 3c74c92:      	movq	%r13, %rbx
 3c74c95:      	subq	%rcx, %rbx
 3c74c98:      	movq	0x10(%rsp), %r15
 3c74c9d:      	cmpq	-0xf0(%rbx), %r15
 3c74ca4:      	jne	0x3c74cbb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x2cb>
 3c74ca6:      	movq	-0xf8(%rbx), %rsi
 3c74cad:      	movq	0x18(%rsp), %rdi
 3c74cb2:      	movq	%r15, %rdx
 3c74cb5:      	callq	*%rbp
 3c74cb7:      	testl	%eax, %eax
 3c74cb9:      	je	0x3c74d0e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x31e>
 3c74cbb:      	leal	-0x1(%r12), %eax
 3c74cc0:      	andw	%r12w, %ax
 3c74cc4:      	movl	%eax, %r12d
 3c74cc7:      	movq	0x30(%rsp), %rax
 3c74ccc:      	movq	0x8(%rsp), %r15
 3c74cd1:      	movdqa	0x60(%rsp), %xmm1
 3c74cd7:      	movq	0x50(%rsp), %rsi
 3c74cdc:      	pcmpeqd	%xmm2, %xmm2
 3c74ce0:      	movq	0x48(%rsp), %rdi
 3c74ce5:      	movdqa	0x70(%rsp), %xmm3
 3c74ceb:      	jne	0x3c74c83 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x293>
 3c74ced:      	pcmpeqb	%xmm2, %xmm3
 3c74cf1:      	pmovmskb	%xmm3, %ecx
 3c74cf5:      	movl	$0xffffffff, %edx       # imm = 0xFFFFFFFF
 3c74cfa:      	testl	%ecx, %ecx
 3c74cfc:      	jne	0x3c74d3b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x34b>
 3c74cfe:      	addq	%rsi, %rax
 3c74d01:      	addq	$0x10, %rax
 3c74d05:      	addq	$0x10, %rsi
 3c74d09:      	jmp	0x3c74c43 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x253>
 3c74d0e:      	movq	-0x8(%rbx), %rax
 3c74d12:      	movq	0x8(%rsp), %rcx
 3c74d17:      	cmpq	%rcx, %rax
 3c74d1a:      	cmovaq	%rax, %rcx
 3c74d1e:      	movl	$0xffffffff, %edx       # imm = 0xFFFFFFFF
 3c74d23:      	movq	%r15, %r12
 3c74d26:      	movq	%rcx, %r15
 3c74d29:      	jmp	0x3c74d40 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x350>
 3c74d2b:      	movl	$0xffffffff, 0x30(%rsp) # imm = 0xFFFFFFFF
 3c74d33:      	xorl	%r13d, %r13d
 3c74d36:      	jmp	0x3c74e6f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47f>
 3c74d3b:      	movq	0x10(%rsp), %r12
 3c74d40:      	movl	%edx, 0x30(%rsp)
 3c74d44:      	leaq	0x28(%r14), %rdi
 3c74d48:      	movq	0x18(%rsp), %rsi
 3c74d4d:      	movq	%r12, %rdx
 3c74d50:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c74d55:      	movq	%rax, %rcx
 3c74d58:      	shrq	$0x39, %rcx
 3c74d5c:      	movq	0x8(%r14), %rsi
 3c74d60:      	movq	0x10(%r14), %rdx
 3c74d64:      	movd	%ecx, %xmm0
 3c74d68:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74d6c:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c74d71:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c74d76:      	xorl	%r13d, %r13d
 3c74d79:      	pcmpeqd	%xmm2, %xmm2
 3c74d7d:      	movq	0x14cc024(%rip), %rbp   # 0x5140da8 <writev+0x5140da8>
 3c74d84:      	xorl	%edi, %edi
 3c74d86:      	movq	0x10(%rsp), %r12
 3c74d8b:      	andq	%rdx, %rax
 3c74d8e:      	movdqu	(%rsi,%rax), %xmm3
 3c74d93:      	movdqa	%xmm3, %xmm0
 3c74d97:      	pcmpeqb	%xmm1, %xmm0
 3c74d9b:      	pmovmskb	%xmm0, %ebx
 3c74d9f:      	testl	%ebx, %ebx
 3c74da1:      	je	0x3c74e43 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x453>
 3c74da7:      	movq	%rax, 0x50(%rsp)
 3c74dac:      	movq	%r15, 0x8(%rsp)
 3c74db1:      	movq	%rdx, 0x48(%rsp)
 3c74db6:      	movdqa	%xmm1, 0x70(%rsp)
 3c74dbc:      	movq	%rsi, 0x90(%rsp)
 3c74dc4:      	movq	%rdi, 0x88(%rsp)
 3c74dcc:      	movdqa	%xmm3, 0xb0(%rsp)
 3c74dd5:      	tzcntl	%ebx, %ecx
 3c74dd9:      	addq	%rax, %rcx
 3c74ddc:      	andq	%rdx, %rcx
 3c74ddf:      	shlq	$0x8, %rcx
 3c74de3:      	movq	%rsi, %r15
 3c74de6:      	subq	%rcx, %r15
 3c74de9:      	cmpq	-0xf0(%r15), %r12
 3c74df0:      	jne	0x3c74e07 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x417>
 3c74df2:      	movq	-0xf8(%r15), %rsi
 3c74df9:      	movq	0x18(%rsp), %rdi
 3c74dfe:      	movq	%r12, %rdx
 3c74e01:      	callq	*%rbp
 3c74e03:      	testl	%eax, %eax
 3c74e05:      	je	0x3c74e5f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x46f>
 3c74e07:      	leal	-0x1(%rbx), %eax
 3c74e0a:      	andw	%bx, %ax
 3c74e0d:      	movl	%eax, %ebx
 3c74e0f:      	movq	0x50(%rsp), %rax
 3c74e14:      	movq	0x8(%rsp), %r15
 3c74e19:      	movq	0x48(%rsp), %rdx
 3c74e1e:      	movdqa	0x70(%rsp), %xmm1
 3c74e24:      	pcmpeqd	%xmm2, %xmm2
 3c74e28:      	movq	0x90(%rsp), %rsi
 3c74e30:      	movq	0x88(%rsp), %rdi
 3c74e38:      	movdqa	0xb0(%rsp), %xmm3
 3c74e41:      	jne	0x3c74dd5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x3e5>
 3c74e43:      	pcmpeqb	%xmm2, %xmm3
 3c74e47:      	pmovmskb	%xmm3, %ecx
 3c74e4b:      	testl	%ecx, %ecx
 3c74e4d:      	jne	0x3c74e6a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47a>
 3c74e4f:      	addq	%rdi, %rax
 3c74e52:      	addq	$0x10, %rax
 3c74e56:      	addq	$0x10, %rdi
 3c74e5a:      	jmp	0x3c74d8b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x39b>
 3c74e5f:      	movq	-0x8(%r15), %r13
 3c74e63:      	movq	0x8(%rsp), %r15
 3c74e68:      	jmp	0x3c74e6f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47f>
 3c74e6a:      	movq	0x10(%rsp), %r12
 3c74e6f:      	movq	0x38(%r14), %rcx
 3c74e73:      	xorl	%eax, %eax
 3c74e75:      	subq	%r13, %rcx
 3c74e78:      	cmovaeq	%rcx, %rax
 3c74e7c:      	movb	$0x2, %bl
 3c74e7e:      	addq	%r15, %rax
 3c74e81:      	jb	0x3c74b96 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x1a6>
 3c74e87:      	cmpq	0x40(%r14), %rax
 3c74e8b:      	movq	0x28(%rsp), %r13
 3c74e90:      	jbe	0x3c74f06 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x516>
 3c74e92:      	cmpb	$0x0, 0x24(%rsp)
 3c74e97:      	jne	0x3c74eb6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c74e99:      	movq	0x14cc1c8(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c74ea0:      	movq	(%rax), %rax
 3c74ea3:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c74ead:      	testq	%rcx, %rax
 3c74eb0:      	jne	0x3c7516d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x77d>
 3c74eb6:      	xorl	%eax, %eax
 3c74eb8:      	xchgl	%eax, (%r14)
 3c74ebb:      	cmpl	$0x2, %eax
 3c74ebe:      	je	0x3c7515f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x76f>
 3c74ec4:      	cmpq	$-0x1, 0x18(%r13)
 3c74ec9:      	je	0x3c74ed4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4e4>
 3c74ecb:      	leaq	0x18(%r13), %rdi
 3c74ecf:      	callq	0x3c08ae0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c74ed4:      	movq	(%r13), %rsi
 3c74ed8:      	cmpq	$-0x1, %rsi
 3c74edc:      	je	0x3c74ef2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c74ede:      	testq	%rsi, %rsi
 3c74ee1:      	je	0x3c74ef2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c74ee3:      	movq	0x8(%r13), %rdi
 3c74ee7:      	movl	$0x1, %edx
 3c74eec:      	callq	*0x14cbe4e(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c74ef2:      	movl	%ebx, %eax
 3c74ef4:      	addq	$0x298, %rsp            # imm = 0x298
 3c74efb:      	popq	%rbx
 3c74efc:      	popq	%r12
 3c74efe:      	popq	%r13
 3c74f00:      	popq	%r14
 3c74f02:      	popq	%r15
 3c74f04:      	popq	%rbp
 3c74f05:      	retq
 3c74f06:      	movq	%rax, 0x38(%r14)
 3c74f0a:      	movq	%r13, %rbx
 3c74f0d:      	testq	%r12, %r12
 3c74f10:      	jns	0x3c74f28 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x538>
 3c74f12:      	xorl	%ebp, %ebp
 3c74f14:      	movq	%rbp, %rdi
 3c74f17:      	movq	%r12, %rsi
 3c74f1a:      	movq	%rbx, %r13
 3c74f1d:      	callq	*0x14cbe65(%rip)        # 0x5140d88 <writev+0x5140d88>
 3c74f23:      	jmp	0x3c7515d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x76d>
 3c74f28:      	je	0x3c74f5e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x56e>
 3c74f2a:      	callq	*0x14cbe60(%rip)        # 0x5140d90 <writev+0x5140d90>
 3c74f30:      	movl	$0x1, %ebp
 3c74f35:      	movl	$0x1, %esi
 3c74f3a:      	movq	%r12, %rdi
 3c74f3d:      	callq	*0x14cbe55(%rip)        # 0x5140d98 <writev+0x5140d98>
 3c74f43:      	testq	%rax, %rax
 3c74f46:      	je	0x3c74f14 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x524>
 3c74f48:      	movq	%rax, %r13
 3c74f4b:      	movq	%rax, %rdi
 3c74f4e:      	movq	0x18(%rsp), %rsi
 3c74f53:      	movq	%r12, %rdx
 3c74f56:      	callq	*0x14cbddc(%rip)        # 0x5140d38 <writev+0x5140d38>
 3c74f5c:      	jmp	0x3c74f64 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x574>
 3c74f5e:      	movl	$0x1, %r13d
 3c74f64:      	movq	%r12, 0x98(%rsp)
 3c74f6c:      	movq	%r13, 0xa0(%rsp)
 3c74f74:      	movq	%r12, 0xa8(%rsp)
 3c74f7c:      	movq	%rbx, %r13
 3c74f7f:      	movq	0x60(%rbx), %rax
 3c74f83:      	movq	%rax, 0x120(%rsp)
 3c74f8b:      	movups	0x50(%rbx), %xmm0
 3c74f8f:      	movaps	%xmm0, 0x110(%rsp)
 3c74f97:      	movups	0x40(%rbx), %xmm0
 3c74f9b:      	movaps	%xmm0, 0x100(%rsp)
 3c74fa3:      	movdqu	(%rbx), %xmm0
 3c74fa7:      	movdqu	0x10(%rbx), %xmm1
 3c74fac:      	movdqu	0x20(%rbx), %xmm2
 3c74fb1:      	movdqu	0x30(%rbx), %xmm3
 3c74fb6:      	movdqa	%xmm3, 0xf0(%rsp)
 3c74fbf:      	movdqa	%xmm2, 0xe0(%rsp)
 3c74fc8:      	movdqa	%xmm1, 0xd0(%rsp)
 3c74fd1:      	movdqa	%xmm0, 0xc0(%rsp)
 3c74fda:      	movq	%r15, 0x1a0(%rsp)
 3c74fe2:      	movq	0x60(%rsp), %rax
 3c74fe7:      	movq	%rax, 0x190(%rsp)
 3c74fef:      	movl	0x30(%rsp), %eax
 3c74ff3:      	movl	%eax, 0x198(%rsp)
 3c74ffa:      	movq	$-0x2, 0x128(%rsp)
 3c75006:      	movl	$0x0, 0x4(%rsp)
 3c7500e:      	leaq	0x1b0(%rsp), %rdi
 3c75016:      	leaq	0x98(%rsp), %rdx
 3c7501e:      	leaq	0xc0(%rsp), %rcx
 3c75026:      	leaq	0x8(%r14), %rsi
 3c7502a:      	callq	0x3e99330 <_RNvMs1_NtCsgQfI1edjipl_9hashbrown3mapINtB5_7HashMapNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateE6insertB1s_>
 3c7502f:      	movq	0x1b0(%rsp), %r12
 3c75037:      	cmpq	$-0x2, %r12
 3c7503b:      	je	0x3c750c1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c75041:      	cmpq	$-0x1, 0x1c8(%rsp)
 3c7504a:      	je	0x3c75059 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x669>
 3c7504c:      	leaq	0x1c8(%rsp), %rdi
 3c75054:      	callq	0x3c08ae0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c75059:      	cmpq	$-0x1, %r12
 3c7505d:      	je	0x3c7507a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x68a>
 3c7505f:      	testq	%r12, %r12
 3c75062:      	je	0x3c7507a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x68a>
 3c75064:      	movq	0x1b8(%rsp), %rdi
 3c7506c:      	movl	$0x1, %edx
 3c75071:      	movq	%r12, %rsi
 3c75074:      	callq	*0x14cbcc6(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c7507a:      	movq	0x218(%rsp), %r12
 3c75082:      	cmpq	$-0x2, %r12
 3c75086:      	je	0x3c750c1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c75088:      	cmpq	$-0x1, 0x230(%rsp)
 3c75091:      	je	0x3c750a0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6b0>
 3c75093:      	leaq	0x230(%rsp), %rdi
 3c7509b:      	callq	0x3c08ae0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c750a0:      	cmpq	$-0x1, %r12
 3c750a4:      	je	0x3c750c1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c750a6:      	testq	%r12, %r12
 3c750a9:      	je	0x3c750c1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c750ab:      	movq	0x220(%rsp), %rdi
 3c750b3:      	movl	$0x1, %edx
 3c750b8:      	movq	%r12, %rsi
 3c750bb:      	callq	*0x14cbc7f(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c750c1:      	cmpb	$0x0, 0x24(%rsp)
 3c750c6:      	jne	0x3c750e5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6f5>
 3c750c8:      	movq	0x14cbf99(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c750cf:      	movq	(%rax), %rax
 3c750d2:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c750dc:      	testq	%rcx, %rax
 3c750df:      	jne	0x3c75185 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x795>
 3c750e5:      	xorl	%eax, %eax
 3c750e7:      	xchgl	%eax, (%r14)
 3c750ea:      	movb	$-0x1, %bl
 3c750ec:      	cmpl	$0x2, %eax
 3c750ef:      	jne	0x3c74ef2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c750f5:      	movq	%r14, %rdi
 3c750f8:      	callq	*0x14cbf72(%rip)        # 0x5141070 <writev+0x5141070>
 3c750fe:      	jmp	0x3c74ef2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c75103:      	movq	%r14, %rdi
 3c75106:      	callq	*0x14cc2fc(%rip)        # 0x5141408 <writev+0x5141408>
 3c7510c:      	jmp	0x3c74a20 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x30>
 3c75111:      	callq	*0x14cbf61(%rip)        # 0x5141078 <writev+0x5141078>
 3c75117:      	xorb	$0x1, %al
 3c75119:      	movzbl	0x4(%r14), %ecx
 3c7511e:      	testb	%cl, %cl
 3c75120:      	je	0x3c74a45 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x55>
 3c75126:      	movq	%r14, 0xc0(%rsp)
 3c7512e:      	movb	%al, 0xc8(%rsp)
 3c75135:      	leaq	-0x2e5675b(%rip), %rdi  # 0xe1e9e1 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c7513c:      	leaq	0x13b73ad(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c75143:      	leaq	0x13b796e(%rip), %r8    # 0x502cab8 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2f8>
 3c7514a:      	leaq	0xc0(%rsp), %rdx
 3c75152:      	movl	$0x2b, %esi
 3c75157:      	callq	*0x14cbd3b(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c7515d:      	ud2
 3c7515f:      	movq	%r14, %rdi
 3c75162:      	callq	*0x14cbf08(%rip)        # 0x5141070 <writev+0x5141070>
 3c75168:      	jmp	0x3c74ec4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4d4>
 3c7516d:      	callq	*0x14cbf05(%rip)        # 0x5141078 <writev+0x5141078>
 3c75173:      	testb	%al, %al
 3c75175:      	jne	0x3c74eb6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c7517b:      	movb	$0x1, 0x4(%r14)
 3c75180:      	jmp	0x3c74eb6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c75185:      	callq	*0x14cbeed(%rip)        # 0x5141078 <writev+0x5141078>
 3c7518b:      	testb	%al, %al
 3c7518d:      	jne	0x3c750e5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6f5>
 3c75193:      	movb	$0x1, 0x4(%r14)
 3c75198:      	jmp	0x3c750e5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6f5>
 3c7519d:      	movq	%rax, %r15
 3c751a0:      	jmp	0x3c75260 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x870>
 3c751a5:      	movq	%rax, %r15
 3c751a8:      	testq	%r12, %r12
 3c751ab:      	jle	0x3c751fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x80b>
 3c751ad:      	movq	0x220(%rsp), %rdi
 3c751b5:      	movl	$0x1, %edx
 3c751ba:      	movq	%r12, %rsi
 3c751bd:      	callq	*0x14cbb7d(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c751c3:      	jmp	0x3c751fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x80b>
 3c751c5:      	movq	%rax, %r15
 3c751c8:      	testq	%r12, %r12
 3c751cb:      	jle	0x3c751e3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x7f3>
 3c751cd:      	movq	0x1b8(%rsp), %rdi
 3c751d5:      	movl	$0x1, %edx
 3c751da:      	movq	%r12, %rsi
 3c751dd:      	callq	*0x14cbb5d(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c751e3:      	cmpq	$-0x2, 0x218(%rsp)
 3c751ec:      	je	0x3c751fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x80b>
 3c751ee:      	leaq	0x218(%rsp), %rdi
 3c751f6:      	callq	0x3c08970 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c751fb:      	movl	$0x0, 0x4(%rsp)
 3c75203:      	movq	%rbx, %r13
 3c75206:      	jmp	0x3c7524c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x85c>
 3c75208:      	callq	*0x14cbbaa(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c7520e:      	movq	%rax, %r15
 3c75211:      	movq	(%r13), %rsi
 3c75215:      	testq	%rsi, %rsi
 3c75218:      	jle	0x3c75268 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x878>
 3c7521a:      	movq	0x8(%r13), %rdi
 3c7521e:      	movl	$0x1, %edx
 3c75223:      	callq	*0x14cbb17(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c75229:      	movq	%r15, %rdi
 3c7522c:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c75231:      	movq	%rax, %r15
 3c75234:      	leaq	0xc0(%rsp), %rdi
 3c7523c:      	callq	0x3c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c75241:      	jmp	0x3c75260 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x870>
 3c75243:      	callq	*0x14cbb6f(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c75249:      	movq	%rax, %r15
 3c7524c:      	movzbl	0x24(%rsp), %esi
 3c75251:      	movq	%r14, %rdi
 3c75254:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c75259:      	cmpb	$0x0, 0x4(%rsp)
 3c7525e:      	je	0x3c75268 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x878>
 3c75260:      	movq	%r13, %rdi
 3c75263:      	callq	0x3c08970 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c75268:      	movq	%r15, %rdi
 3c7526b:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c75270:      	callq	*0x14cbb42(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c75276:      	int3
 3c75277:      	int3
 3c75278:      	int3
 3c75279:      	int3
 3c7527a:      	int3
 3c7527b:      	int3
 3c7527c:      	int3
 3c7527d:      	int3
 3c7527e:      	int3
 3c7527f:      	int3

0000000003c75280 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take>:
 3c75280:      	pushq	%rbp
 3c75281:      	pushq	%r15
 3c75283:      	pushq	%r14
 3c75285:      	pushq	%r13
 3c75287:      	pushq	%r12
 3c75289:      	pushq	%rbx
 3c7528a:      	subq	$0x238, %rsp            # imm = 0x238
 3c75291:      	movq	%rcx, %r15
 3c75294:      	movq	%rdx, %r12
 3c75297:      	movq	%rsi, %rbx
 3c7529a:      	movq	%rdi, %r13
 3c7529d:      	movl	$0x1, %ecx
 3c752a2:      	xorl	%eax, %eax
 3c752a4:      	lock
 3c752a5:      	cmpxchgl	%ecx, (%rsi)
 3c752a8:      	jne	0x3c755bf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x33f>
 3c752ae:      	movq	0x14cbdb3(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c752b5:      	movq	(%r14), %rax
 3c752b8:      	shlq	%rax
 3c752bb:      	testq	%rax, %rax
 3c752be:      	jne	0x3c755cd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x34d>
 3c752c4:      	xorl	%eax, %eax
 3c752c6:      	movzbl	0x4(%rbx), %ecx
 3c752ca:      	testb	%cl, %cl
 3c752cc:      	jne	0x3c755e1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x361>
 3c752d2:      	movl	%eax, 0x4(%rsp)
 3c752d6:      	leaq	0x8(%rbx), %rsi
 3c752da:      	movq	%rbx, %rdi
 3c752dd:      	callq	0x3c47600 <_RNvMNtCs3pwlnhBXFtN_12memra_server9job_storeNtB2_16InMemoryJobStore12sweep_locked.llvm.12140772379724168186>
 3c752e2:      	cmpq	$0x0, 0x20(%rbx)
 3c752e7:      	je	0x3c7541a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x19a>
 3c752ed:      	leaq	0x28(%rbx), %rdi
 3c752f1:      	movq	%r12, %rsi
 3c752f4:      	movq	%r15, %rdx
 3c752f7:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c752fc:      	movq	%rax, %rbp
 3c752ff:      	shrq	$0x39, %rax
 3c75303:      	movq	0x8(%rbx), %rcx
 3c75307:      	movq	0x10(%rbx), %rdx
 3c7530b:      	movd	%eax, %xmm0
 3c7530f:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c75313:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c75318:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c7531d:      	xorl	%esi, %esi
 3c7531f:      	pcmpeqd	%xmm2, %xmm2
 3c75323:      	movq	0x14cba7e(%rip), %r8    # 0x5140da8 <writev+0x5140da8>
 3c7532a:      	andq	%rdx, %rbp
 3c7532d:      	movdqu	(%rcx,%rbp), %xmm3
 3c75332:      	movdqa	%xmm3, %xmm0
 3c75336:      	pcmpeqb	%xmm1, %xmm0
 3c7533a:      	pmovmskb	%xmm0, %eax
 3c7533e:      	testl	%eax, %eax
 3c75340:      	je	0x3c753e0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x160>
 3c75346:      	movq	%r13, 0x8(%rsp)
 3c7534b:      	movq	%rcx, 0x20(%rsp)
 3c75350:      	movq	%rdx, 0x18(%rsp)
 3c75355:      	movdqa	%xmm1, 0x40(%rsp)
 3c7535b:      	movq	%rsi, 0x10(%rsp)
 3c75360:      	movdqa	%xmm3, 0x30(%rsp)
 3c75366:      	movq	%rax, 0x28(%rsp)
 3c7536b:      	tzcntl	%eax, %eax
 3c7536f:      	addq	%rbp, %rax
 3c75372:      	andq	%rdx, %rax
 3c75375:      	shlq	$0x8, %rax
 3c75379:      	movq	%rcx, %r13
 3c7537c:      	subq	%rax, %r13
 3c7537f:      	cmpq	-0xf0(%r13), %r15
 3c75386:      	jne	0x3c753ae <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x12e>
 3c75388:      	movq	-0xf8(%r13), %rsi
 3c7538f:      	movq	%r12, %rdi
 3c75392:      	movq	%r15, %rdx
 3c75395:      	movq	%r15, %r14
 3c75398:      	movq	%r12, %r15
 3c7539b:      	movq	%r8, %r12
 3c7539e:      	callq	*%r8
 3c753a1:      	movq	%r12, %r8
 3c753a4:      	movq	%r15, %r12
 3c753a7:      	movq	%r14, %r15
 3c753aa:      	testl	%eax, %eax
 3c753ac:      	je	0x3c75403 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x183>
 3c753ae:      	movq	0x28(%rsp), %rcx
 3c753b3:      	leal	-0x1(%rcx), %eax
 3c753b6:      	andw	%cx, %ax
 3c753b9:      	movq	0x8(%rsp), %r13
 3c753be:      	movq	0x20(%rsp), %rcx
 3c753c3:      	movq	0x18(%rsp), %rdx
 3c753c8:      	movdqa	0x40(%rsp), %xmm1
 3c753ce:      	movq	0x10(%rsp), %rsi
 3c753d3:      	pcmpeqd	%xmm2, %xmm2
 3c753d7:      	movdqa	0x30(%rsp), %xmm3
 3c753dd:      	jne	0x3c75366 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0xe6>
 3c753df:      	nop
 3c753e0:      	pcmpeqb	%xmm2, %xmm3
 3c753e4:      	pmovmskb	%xmm3, %eax
 3c753e8:      	testl	%eax, %eax
 3c753ea:      	movq	0x14cbc77(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c753f1:      	jne	0x3c7541a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x19a>
 3c753f3:      	addq	%rsi, %rbp
 3c753f6:      	addq	$0x10, %rbp
 3c753fa:      	addq	$0x10, %rsi
 3c753fe:      	jmp	0x3c7532a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0xaa>
 3c75403:      	cmpq	$-0x2, -0x80(%r13)
 3c75408:      	movq	0x8(%rsp), %r13
 3c7540d:      	movq	0x14cbc54(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c75414:      	jne	0x3c75580 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x300>
 3c7541a:      	leaq	0x28(%rbx), %rdi
 3c7541e:      	movq	%r12, %rsi
 3c75421:      	movq	%r15, %rdx
 3c75424:      	callq	0x3e68ec0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c75429:      	leaq	0x58(%rsp), %rdi
 3c7542e:      	leaq	0x8(%rbx), %rsi
 3c75432:      	movq	%rax, %rdx
 3c75435:      	movq	%r12, %rcx
 3c75438:      	movq	%r15, %r8
 3c7543b:      	callq	0x389afa0 <_RINvMs6_NtCsgQfI1edjipl_9hashbrown3rawINtB6_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryEE12remove_entryNCINvNtB8_3map14equivalent_keyeBQ_B1r_E0EB1v_>
 3c75440:      	movq	0x58(%rsp), %r15
 3c75445:      	cmpq	$-0x1, %r15
 3c75449:      	je	0x3c7556d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x2ed>
 3c7544f:      	movq	0x70(%rsp), %r12
 3c75454:      	leaq	0x78(%rsp), %rsi
 3c75459:      	leaq	0x158(%rsp), %rdi
 3c75461:      	movl	$0xe0, %edx
 3c75466:      	callq	*0x14cb8cc(%rip)        # 0x5140d38 <writev+0x5140d38>
 3c7546c:      	testq	%r15, %r15
 3c7546f:      	je	0x3c75484 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x204>
 3c75471:      	movq	0x60(%rsp), %rdi
 3c75476:      	movl	$0x1, %edx
 3c7547b:      	movq	%r15, %rsi
 3c7547e:      	callq	*0x14cb8bc(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c75484:      	cmpq	$-0x2, %r12
 3c75488:      	movl	0x4(%rsp), %ebp
 3c7548c:      	je	0x3c75571 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x2f1>
 3c75492:      	movq	%r12, 0x58(%rsp)
 3c75497:      	leaq	0x60(%rsp), %rdi
 3c7549c:      	leaq	0x158(%rsp), %rsi
 3c754a4:      	movl	$0xe0, %edx
 3c754a9:      	callq	*0x14cb889(%rip)        # 0x5140d38 <writev+0x5140d38>
 3c754af:      	movq	0x38(%rbx), %rax
 3c754b3:      	xorl	%ecx, %ecx
 3c754b5:      	subq	0x138(%rsp), %rax
 3c754bd:      	cmovaeq	%rax, %rcx
 3c754c1:      	movq	%rcx, 0x38(%rbx)
 3c754c5:      	movq	0xc0(%rsp), %r15
 3c754cd:      	cmpq	$-0x2, %r15
 3c754d1:      	je	0x3c7550c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x28c>
 3c754d3:      	cmpq	$-0x1, 0xd8(%rsp)
 3c754dc:      	je	0x3c754eb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x26b>
 3c754de:      	leaq	0xd8(%rsp), %rdi
 3c754e6:      	callq	0x3c08ae0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c754eb:      	cmpq	$-0x1, %r15
 3c754ef:      	je	0x3c7550c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x28c>
 3c754f1:      	testq	%r15, %r15
 3c754f4:      	je	0x3c7550c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x28c>
 3c754f6:      	movq	0xc8(%rsp), %rdi
 3c754fe:      	movl	$0x1, %edx
 3c75503:      	movq	%r15, %rsi
 3c75506:      	callq	*0x14cb834(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c7550c:      	movq	%r12, (%r13)
 3c75510:      	movups	0x158(%rsp), %xmm0
 3c75518:      	movdqu	0x168(%rsp), %xmm1
 3c75521:      	movdqu	0x178(%rsp), %xmm2
 3c7552a:      	movdqu	0x188(%rsp), %xmm3
 3c75533:      	movups	%xmm0, 0x8(%r13)
 3c75538:      	movdqu	%xmm1, 0x18(%r13)
 3c7553e:      	movdqu	%xmm2, 0x28(%r13)
 3c75544:      	movdqu	%xmm3, 0x38(%r13)
 3c7554a:      	movups	0x198(%rsp), %xmm0
 3c75552:      	movups	%xmm0, 0x48(%r13)
 3c75557:      	movdqu	0x1a8(%rsp), %xmm0
 3c75560:      	movdqu	%xmm0, 0x58(%r13)
 3c75566:      	testb	%bpl, %bpl
 3c75569:      	je	0x3c7558f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x30f>
 3c7556b:      	jmp	0x3c755a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c7556d:      	movl	0x4(%rsp), %ebp
 3c75571:      	movq	$-0x2, (%r13)
 3c75579:      	testb	%bpl, %bpl
 3c7557c:      	je	0x3c7558f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x30f>
 3c7557e:      	jmp	0x3c755a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c75580:      	movq	$-0x2, (%r13)
 3c75588:      	cmpb	$0x0, 0x4(%rsp)
 3c7558d:      	jne	0x3c755a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c7558f:      	movq	(%r14), %rax
 3c75592:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c7559c:      	testq	%rcx, %rax
 3c7559f:      	jne	0x3c7561c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x39c>
 3c755a1:      	xorl	%eax, %eax
 3c755a3:      	xchgl	%eax, (%rbx)
 3c755a5:      	cmpl	$0x2, %eax
 3c755a8:      	je	0x3c75611 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x391>
 3c755aa:      	movq	%r13, %rax
 3c755ad:      	addq	$0x238, %rsp            # imm = 0x238
 3c755b4:      	popq	%rbx
 3c755b5:      	popq	%r12
 3c755b7:      	popq	%r13
 3c755b9:      	popq	%r14
 3c755bb:      	popq	%r15
 3c755bd:      	popq	%rbp
 3c755be:      	retq
 3c755bf:      	movq	%rbx, %rdi
 3c755c2:      	callq	*0x14cbe40(%rip)        # 0x5141408 <writev+0x5141408>
 3c755c8:      	jmp	0x3c752ae <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x2e>
 3c755cd:      	callq	*0x14cbaa5(%rip)        # 0x5141078 <writev+0x5141078>
 3c755d3:      	xorb	$0x1, %al
 3c755d5:      	movzbl	0x4(%rbx), %ecx
 3c755d9:      	testb	%cl, %cl
 3c755db:      	je	0x3c752d2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x52>
 3c755e1:      	movq	%rbx, 0x58(%rsp)
 3c755e6:      	movb	%al, 0x60(%rsp)
 3c755ea:      	leaq	-0x2e56c10(%rip), %rdi  # 0xe1e9e1 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c755f1:      	leaq	0x13b6ef8(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c755f8:      	leaq	0x13b74d1(%rip), %r8    # 0x502cad0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x310>
 3c755ff:      	leaq	0x58(%rsp), %rdx
 3c75604:      	movl	$0x2b, %esi
 3c75609:      	callq	*0x14cb889(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c7560f:      	ud2
 3c75611:      	movq	%rbx, %rdi
 3c75614:      	callq	*0x14cba56(%rip)        # 0x5141070 <writev+0x5141070>
 3c7561a:      	jmp	0x3c755aa <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x32a>
 3c7561c:      	callq	*0x14cba56(%rip)        # 0x5141078 <writev+0x5141078>
 3c75622:      	testb	%al, %al
 3c75624:      	jne	0x3c755a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c7562a:      	movb	$0x1, 0x4(%rbx)
 3c7562e:      	jmp	0x3c755a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c75633:      	movq	%rax, %r14
 3c75636:      	testq	%r15, %r15
 3c75639:      	jle	0x3c75656 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x3d6>
 3c7563b:      	movq	0xc8(%rsp), %rdi
 3c75643:      	movl	$0x1, %edx
 3c75648:      	movq	%r15, %rsi
 3c7564b:      	callq	*0x14cb6ef(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c75651:      	jmp	0x3c75656 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x3d6>
 3c75653:      	movq	%rax, %r14
 3c75656:      	movzbl	0x4(%rsp), %esi
 3c7565b:      	movq	%rbx, %rdi
 3c7565e:      	callq	0x3c04740 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c75663:      	jmp	0x3c75678 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x3f8>
 3c75665:      	callq	*0x14cb74d(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c7566b:      	movq	%rax, %r14
 3c7566e:      	leaq	0x58(%rsp), %rdi
 3c75673:      	callq	0x3c03360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c75678:      	movq	%r14, %rdi
 3c7567b:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c75680:      	callq	*0x14cb732(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c75686:      	int3
 3c75687:      	int3
 3c75688:      	int3
 3c75689:      	int3
 3c7568a:      	int3
 3c7568b:      	int3
 3c7568c:      	int3
 3c7568d:      	int3
 3c7568e:      	int3
 3c7568f:      	int3

0000000003ce3f60 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>:
 3ce3f60:      	pushq	%rbp
 3ce3f61:      	pushq	%r15
 3ce3f63:      	pushq	%r14
 3ce3f65:      	pushq	%r13
 3ce3f67:      	pushq	%r12
 3ce3f69:      	pushq	%rbx
 3ce3f6a:      	subq	$0xe78, %rsp            # imm = 0xE78
 3ce3f71:      	movq	%r8, %r14
 3ce3f74:      	movl	%edx, 0x24(%rsp)
 3ce3f78:      	movq	%r9, 0xa8(%rsp)
 3ce3f80:      	movq	0x18(%rcx), %rbp
 3ce3f84:      	testq	%rbp, %rbp
 3ce3f87:      	movq	%rdi, 0x30(%rsp)
 3ce3f8c:      	je	0x3ce4392 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x432>
 3ce3f92:      	movq	0x10(%r14), %rax
 3ce3f96:      	movq	%rax, 0x70(%rsp)
 3ce3f9b:      	movups	(%r14), %xmm0
 3ce3f9f:      	movaps	%xmm0, 0x60(%rsp)
 3ce3fa4:      	movq	%r9, 0xb0(%rsp)
 3ce3fac:      	movq	0x28(%rbp), %r14
 3ce3fb0:      	movl	0x146a12a(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x10>
 3ce3fb6:      	testl	%eax, %eax
 3ce3fb8:      	jne	0x3ce52cf <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x136f>
 3ce3fbe:      	leaq	0x10(%rbp), %rax
 3ce3fc2:      	movq	%rax, 0x40(%rsp)
 3ce3fc7:      	movzbl	0x146a102(%rip), %ecx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 3ce3fce:      	movq	0x146a103(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x8>
 3ce3fd5:      	movq	%r14, %rdx
 3ce3fd8:      	shrq	$0x3e, %rdx
 3ce3fdc:      	movzbl	0x24(%rsp), %ebx
 3ce3fe1:      	jne	0x3ce52e0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1380>
 3ce3fe7:      	leaq	(,%r14,4), %rdx
 3ce3fef:      	testb	%cl, %cl
 3ce3ff1:      	cmovneq	%rax, %rdx
 3ce3ff5:      	movq	%rdx, 0x48(%rsp)
 3ce3ffa:      	testq	%r14, %r14
 3ce3ffd:      	je	0x3ce5356 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13f6>
 3ce4003:      	movq	%rbx, 0x38(%rsp)
 3ce4008:      	nopl	(%rax,%rax)
 3ce4010:      	movq	0x30(%rbp,%rbx,8), %rax
 3ce4015:      	movq	%rax, 0x50(%rsp)
 3ce401a:      	movq	0x30(%rbp), %rbx
 3ce401e:      	movq	0x38(%rbp), %r12
 3ce4022:      	movq	0x40(%rbp), %r15
 3ce4026:      	movl	0x146a0e4(%rip), %eax   # 0x514e110 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451+0x8>
 3ce402c:      	testl	%eax, %eax
 3ce402e:      	jne	0x3ce418a <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x22a>
 3ce4034:      	addq	%rbx, %r12
 3ce4037:      	addq	%r15, %r12
 3ce403a:      	movq	0x146a0c7(%rip), %rsi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451>
 3ce4041:      	leaq	0x10(%rbp), %rdi
 3ce4045:      	callq	0x39b5bb0 <_RNvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB4_9RouteLoad18service_estimate_s>
 3ce404a:      	movq	%rax, %rcx
 3ce404d:      	movq	%r12, %rax
 3ce4050:      	orq	%r14, %rax
 3ce4053:      	shrq	$0x20, %rax
 3ce4057:      	je	0x3ce4150 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1f0>
 3ce405d:      	movq	%r12, %rax
 3ce4060:      	xorl	%edx, %edx
 3ce4062:      	divq	%r14
 3ce4065:      	movq	%rax, %rdx
 3ce4068:      	incq	%rdx
 3ce406b:      	movq	%rcx, %rax
 3ce406e:      	mulq	%rdx
 3ce4071:      	jo	0x3ce4169 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x209>
 3ce4077:      	movq	%rax, %r13
 3ce407a:      	movq	%r13, 0x28(%rsp)
 3ce407f:      	movq	0x50(%rsp), %r15
 3ce4084:      	cmpq	0x48(%rsp), %r15
 3ce4089:      	jae	0x3ce480b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x8ab>
 3ce408f:      	movq	0x60(%rbp), %rbx
 3ce4093:      	callq	*0x145e047(%rip)        # 0x51420e0 <writev+0x51420e0>
 3ce4099:      	leaq	0x68(%rsp), %rdi
 3ce409e:      	movq	%rax, %rsi
 3ce40a1:      	callq	*0x145e049(%rip)        # 0x51420f0 <writev+0x51420f0>
 3ce40a7:      	imulq	$0x3e8, %rax, %rax      # imm = 0x3E8
 3ce40ae:      	movl	%edx, %ecx
 3ce40b0:      	imulq	$0x431bde83, %rcx, %rcx # imm = 0x431BDE83
 3ce40b7:      	shrq	$0x32, %rcx
 3ce40bb:      	addq	%rax, %rcx
 3ce40be:      	movq	%rcx, 0x58(%rsp)
 3ce40c3:      	cmpb	$0x0, 0x24(%rsp)
 3ce40c8:      	jne	0x3ce4102 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1a2>
 3ce40ca:      	addq	%r12, %rbx
 3ce40cd:      	cmpq	%r14, %rbx
 3ce40d0:      	jb	0x3ce4102 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1a2>
 3ce40d2:      	movq	%r13, %rax
 3ce40d5:      	movl	$0x3e8, %edx            # imm = 0x3E8
 3ce40da:      	mulq	%rdx
 3ce40dd:      	jo	0x3ce419b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x23b>
 3ce40e3:      	cmpq	%rcx, %rax
 3ce40e6:      	ja	0x3ce41ab <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x24b>
 3ce40ec:      	movq	0xb0(%rsp), %rax
 3ce40f4:      	testq	%rax, %rax
 3ce40f7:      	je	0x3ce4102 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1a2>
 3ce40f9:      	cmpq	%rax, %r13
 3ce40fc:      	ja	0x3ce4d1c <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xdbc>
 3ce4102:      	leaq	0x1(%r15), %rcx
 3ce4106:      	movq	%r15, %rax
 3ce4109:      	movq	0x38(%rsp), %rbx
 3ce410e:      	lock
 3ce410f:      	cmpxchgq	%rcx, 0x30(%rbp,%rbx,8)
 3ce4115:      	jne	0x3ce4010 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xb0>
 3ce411b:      	lock
 3ce411c:      	incq	(%rbp)
 3ce4120:      	jle	0x3ce5354 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13f4>
 3ce4126:      	movq	%rbp, 0xc0(%rsp)
 3ce412e:      	callq	*0x145cd7c(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3ce4134:      	cmpl	$-0x1, %edx
 3ce4137:      	je	0x3ce4010 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xb0>
 3ce413d:      	jmp	0x3ce4cc7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xd67>
 3ce4142:      	nopw	%cs:(%rax,%rax)
 3ce4150:      	movl	%r12d, %eax
 3ce4153:      	xorl	%edx, %edx
 3ce4155:      	divl	%r14d
 3ce4158:      	movl	%eax, %edx
 3ce415a:      	incq	%rdx
 3ce415d:      	movq	%rcx, %rax
 3ce4160:      	mulq	%rdx
 3ce4163:      	jno	0x3ce4077 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x117>
 3ce4169:      	movq	$-0x1, %r13
 3ce4170:      	movq	%r13, 0x28(%rsp)
 3ce4175:      	movq	0x50(%rsp), %r15
 3ce417a:      	cmpq	0x48(%rsp), %r15
 3ce417f:      	jb	0x3ce408f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x12f>
 3ce4185:      	jmp	0x3ce480b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x8ab>
 3ce418a:      	leaq	0x1469f77(%rip), %rdi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451>
 3ce4191:      	callq	0x3b2c922 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s0E0zEB1w_>
 3ce4196:      	jmp	0x3ce4034 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xd4>
 3ce419b:      	movq	$-0x1, %rax
 3ce41a2:      	cmpq	%rcx, %rax
 3ce41a5:      	jbe	0x3ce40ec <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x18c>
 3ce41ab:      	movups	0x18(%rbp), %xmm0
 3ce41af:      	movups	%xmm0, 0x5f0(%rsp)
 3ce41b7:      	leaq	0x28(%rsp), %rax
 3ce41bc:      	movq	%rax, 0xc0(%rsp)
 3ce41c4:      	movq	0x145ccf5(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce41cb:      	movq	%rax, 0xc8(%rsp)
 3ce41d3:      	leaq	0x5f0(%rsp), %rcx
 3ce41db:      	movq	%rcx, 0xd0(%rsp)
 3ce41e3:      	leaq	-0x2de70a(%rip), %rcx   # 0x3a05ae0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtReNtB6_5Debug3fmtCs3pwlnhBXFtN_12memra_server>
 3ce41ea:      	movq	%rcx, 0xd8(%rsp)
 3ce41f2:      	leaq	0x58(%rsp), %rcx
 3ce41f7:      	movq	%rcx, 0xe0(%rsp)
 3ce41ff:      	movq	%rax, 0xe8(%rsp)
 3ce4207:      	leaq	-0x2ec3440(%rip), %rsi  # 0xe20dce <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x123>
 3ce420e:      	leaq	0x78(%rsp), %rdi
 3ce4213:      	leaq	0xc0(%rsp), %rdx
 3ce421b:      	callq	*0x145cb37(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4221:      	movq	0x28(%rsp), %rbx
 3ce4226:      	movq	0x80(%rsp), %r14
 3ce422e:      	movq	0x88(%rsp), %rdx
 3ce4236:      	leaq	-0x2ec338f(%rip), %r15  # 0xe20eae <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x203>
 3ce423d:      	movq	%r15, 0x8(%rsp)
 3ce4242:      	movq	$0xd, 0x10(%rsp)
 3ce424b:      	leaq	-0x395d412(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.15708504647765068451>
 3ce4252:      	leaq	0xc0(%rsp), %rdi
 3ce425a:      	movl	$0x10, %r8d
 3ce4260:      	movq	%r14, %rsi
 3ce4263:      	xorl	%r9d, %r9d
 3ce4266:      	callq	0x3cbb480 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce426b:      	movq	0x100(%rsp), %rax
 3ce4273:      	movq	%rax, 0xa70(%rsp)
 3ce427b:      	movups	0xc0(%rsp), %xmm0
 3ce4283:      	movups	0xd0(%rsp), %xmm1
 3ce428b:      	movups	0xe0(%rsp), %xmm2
 3ce4293:      	movups	0xf0(%rsp), %xmm3
 3ce429b:      	movaps	%xmm3, 0xa60(%rsp)
 3ce42a3:      	movaps	%xmm2, 0xa50(%rsp)
 3ce42ab:      	movaps	%xmm1, 0xa40(%rsp)
 3ce42b3:      	movaps	%xmm0, 0xa30(%rsp)
 3ce42bb:      	leaq	0xc0(%rsp), %rdi
 3ce42c3:      	leaq	0xa30(%rsp), %rsi
 3ce42cb:      	callq	0x3eb0090 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce42d0:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce42da:      	leaq	0x5f0(%rsp), %rdi
 3ce42e2:      	leaq	0xc0(%rsp), %rsi
 3ce42ea:      	movl	$0x1, %edx
 3ce42ef:      	movq	%rbx, %rcx
 3ce42f2:      	callq	0x3ce1c70 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce42f7:      	movups	0x660(%rsp), %xmm0
 3ce42ff:      	movq	0x30(%rsp), %r12
 3ce4304:      	movups	%xmm0, 0x70(%r12)
 3ce430a:      	movups	0x650(%rsp), %xmm0
 3ce4312:      	movups	%xmm0, 0x60(%r12)
 3ce4318:      	movups	0x640(%rsp), %xmm0
 3ce4320:      	movups	%xmm0, 0x50(%r12)
 3ce4326:      	movups	0x630(%rsp), %xmm0
 3ce432e:      	movups	%xmm0, 0x40(%r12)
 3ce4334:      	movups	0x5f0(%rsp), %xmm0
 3ce433c:      	movups	0x600(%rsp), %xmm1
 3ce4344:      	movups	0x610(%rsp), %xmm2
 3ce434c:      	movups	0x620(%rsp), %xmm3
 3ce4354:      	movups	%xmm3, 0x30(%r12)
 3ce435a:      	movups	%xmm2, 0x20(%r12)
 3ce4360:      	movups	%xmm1, 0x10(%r12)
 3ce4366:      	movups	%xmm0, (%r12)
 3ce436b:      	movq	%r15, 0x80(%r12)
 3ce4373:      	movq	$0xd, 0x88(%r12)
 3ce437f:      	movq	0x78(%rsp), %rsi
 3ce4384:      	testq	%rsi, %rsi
 3ce4387:      	jne	0x3ce4f01 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfa1>
 3ce438d:      	jmp	0x3ce50f0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4392:      	movl	0x1469df0(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451+0x18>
 3ce4398:      	testl	%eax, %eax
 3ce439a:      	jne	0x3ce52f7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1397>
 3ce43a0:      	movzbl	0x24(%rsp), %r12d
 3ce43a6:      	leaq	0x1469dc3(%rip), %rax   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 3ce43ad:      	movq	(%rax,%r12,8), %rbx
 3ce43b1:      	cmpq	$0x1, %rbx
 3ce43b5:      	movq	%rbx, %r13
 3ce43b8:      	adcq	$0x0, %r13
 3ce43bc:      	movl	0x1469d1e(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x10>
 3ce43c2:      	testl	%eax, %eax
 3ce43c4:      	jne	0x3ce5314 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13b4>
 3ce43ca:      	movq	0x1469cff(%rip), %rdx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 3ce43d1:      	movq	0x1469d00(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x8>
 3ce43d8:      	shrq	$0x3e, %rbx
 3ce43dc:      	jne	0x3ce5348 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13e8>
 3ce43e2:      	leaq	(,%r13,4), %rdi
 3ce43ea:      	testq	%rdx, %rdx
 3ce43ed:      	cmovneq	%rax, %rdi
 3ce43f1:      	movq	%rdi, 0x28(%rsp)
 3ce43f6:      	movq	0x40(%rsi), %rax
 3ce43fa:      	leaq	0x10(%rax), %r15
 3ce43fe:      	addq	$0x8, %r14
 3ce4402:      	movq	0x8(%rcx), %rcx
 3ce4406:      	movq	%rcx, 0x40(%rsp)
 3ce440b:      	movq	%rax, 0x38(%rsp)
 3ce4410:      	addq	$0x18, %rax
 3ce4414:      	movq	%rax, 0xb8(%rsp)
 3ce441c:      	leaq	0x5f0(%rsp), %rbp
 3ce4424:      	xorl	%eax, %eax
 3ce4426:      	movl	$0x1, %ecx
 3ce442b:      	lock
 3ce442c:      	cmpxchgl	%ecx, (%r15)
 3ce4430:      	jne	0x3ce448b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x52b>
 3ce4432:      	nopw	%cs:(%rax,%rax)
 3ce4440:      	movq	0x145cc21(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3ce4447:      	movq	(%rax), %rax
 3ce444a:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3ce4454:      	testq	%rcx, %rax
 3ce4457:      	jne	0x3ce4496 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x536>
 3ce4459:      	xorl	%ebx, %ebx
 3ce445b:      	movq	0x38(%rsp), %rax
 3ce4460:      	movzbl	0x14(%rax), %eax
 3ce4464:      	testb	%al, %al
 3ce4466:      	je	0x3ce44b0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x550>
 3ce4468:      	movq	%r15, 0x5f8(%rsp)
 3ce4470:      	movb	%bl, 0x600(%rsp)
 3ce4477:      	movq	$0x0, 0x5f0(%rsp)
 3ce4483:      	movq	%rbp, %rbx
 3ce4486:      	jmp	0x3ce450d <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x5ad>
 3ce448b:      	movq	%r15, %rdi
 3ce448e:      	callq	*0x145cf74(%rip)        # 0x5141408 <writev+0x5141408>
 3ce4494:      	jmp	0x3ce4440 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x4e0>
 3ce4496:      	callq	*0x145cbdc(%rip)        # 0x5141078 <writev+0x5141078>
 3ce449c:      	movl	%eax, %ebx
 3ce449e:      	xorb	$0x1, %bl
 3ce44a1:      	movq	0x38(%rsp), %rax
 3ce44a6:      	movzbl	0x14(%rax), %eax
 3ce44aa:      	testb	%al, %al
 3ce44ac:      	jne	0x3ce4468 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x508>
 3ce44ae:      	nop
 3ce44b0:      	leaq	0xc0(%rsp), %rdi
 3ce44b8:      	movq	0xb8(%rsp), %rsi
 3ce44c0:      	callq	0x3cf9af0 <_RNvXs1M_NtCs3pwlnhBXFtN_12memra_server6workerNtB6_7MetricsNtNtCs4NRVxsYgnAr_4core5clone5Clone5clone>
 3ce44c5:      	testb	%bl, %bl
 3ce44c7:      	jne	0x3ce44e6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x586>
 3ce44c9:      	movq	0x145cb98(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3ce44d0:      	movq	(%rax), %rax
 3ce44d3:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3ce44dd:      	testq	%rcx, %rax
 3ce44e0:      	jne	0x3ce47ef <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x88f>
 3ce44e6:      	xorl	%eax, %eax
 3ce44e8:      	xchgl	%eax, (%r15)
 3ce44eb:      	cmpl	$0x2, %eax
 3ce44ee:      	movq	%rbp, %rbx
 3ce44f1:      	je	0x3ce47e1 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x881>
 3ce44f7:      	movl	$0x440, %edx            # imm = 0x440
 3ce44fc:      	movq	%rbx, %rdi
 3ce44ff:      	leaq	0xc0(%rsp), %rsi
 3ce4507:      	callq	*0x145c82b(%rip)        # 0x5140d38 <writev+0x5140d38>
 3ce450d:      	leaq	0xa30(%rsp), %rdi
 3ce4515:      	movq	%rbx, %rsi
 3ce4518:      	callq	0x3cf3ba0 <_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtB1v_5mutex10MutexGuardBH_EEE17unwrap_or_defaultBL_>
 3ce451d:      	movq	0xeb0(%rsp), %rax
 3ce4525:      	movq	(%rax,%r12,8), %rbx
 3ce4529:      	movq	%rbx, 0x58(%rsp)
 3ce452e:      	movq	0xb88(%rsp), %rax
 3ce4536:      	testq	%rax, %rax
 3ce4539:      	je	0x3ce4600 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6a0>
 3ce453f:      	movss	0xe68(%rsp), %xmm0
 3ce4548:      	xorps	%xmm1, %xmm1
 3ce454b:      	ucomiss	%xmm1, %xmm0
 3ce454e:      	jbe	0x3ce4600 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6a0>
 3ce4554:      	movsd	0xb90(%rsp), %xmm1
 3ce455d:      	movsd	-0x3963c75(%rip), %xmm3 # 0x3808f0 <anon.78da417e2a41163eb606632dc77243d1.14.llvm.14221926250047164484+0x60>
 3ce4565:      	unpcklps	%xmm3, %xmm1            # xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
 3ce4568:      	movapd	-0x3965570(%rip), %xmm4 # 0x37f000 <writev+0x37f000>
 3ce4570:      	subpd	%xmm4, %xmm1
 3ce4574:      	movapd	%xmm1, %xmm2
 3ce4578:      	unpckhpd	%xmm1, %xmm2            # xmm2 = xmm2[1],xmm1[1]
 3ce457c:      	addsd	%xmm1, %xmm2
 3ce4580:      	movq	%rax, %xmm1
 3ce4585:      	punpckldq	%xmm3, %xmm1    # xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
 3ce4589:      	subpd	%xmm4, %xmm1
 3ce458d:      	movapd	%xmm1, %xmm3
 3ce4591:      	unpckhpd	%xmm1, %xmm3            # xmm3 = xmm3[1],xmm1[1]
 3ce4595:      	addsd	%xmm1, %xmm3
 3ce4599:      	divsd	%xmm3, %xmm2
 3ce459d:      	cvtss2sd	%xmm0, %xmm0
 3ce45a1:      	mulsd	%xmm2, %xmm0
 3ce45a5:      	divsd	-0x395838d(%rip), %xmm0 # 0x38c220 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.580.llvm.6417388326552730234+0x20>
 3ce45ad:      	callq	*0x145cd45(%rip)        # 0x51412f8 <writev+0x51412f8>
 3ce45b3:      	movapd	%xmm0, %xmm1
 3ce45b7:      	subsd	-0x395a137(%rip), %xmm1 # 0x38a488 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.1436.llvm.6417388326552730234+0x40>
 3ce45bf:      	xorpd	%xmm2, %xmm2
 3ce45c3:      	ucomisd	%xmm2, %xmm0
 3ce45c7:      	jb	0x3ce4760 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x800>
 3ce45cd:      	cvttsd2si	%xmm0, %rax
 3ce45d2:      	movq	%rax, %rdx
 3ce45d5:      	sarq	$0x3f, %rdx
 3ce45d9:      	cvttsd2si	%xmm1, %rcx
 3ce45de:      	andq	%rdx, %rcx
 3ce45e1:      	orq	%rax, %rcx
 3ce45e4:      	ucomisd	-0x395ba9c(%rip), %xmm0 # 0x388b50 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.1382.llvm.6417388326552730234+0x28>
 3ce45ec:      	movq	$-0x1, %rax
 3ce45f3:      	jbe	0x3ce4773 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x813>
 3ce45f9:      	jmp	0x3ce4776 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x816>
 3ce45fe:      	nop
 3ce4600:      	movl	0x1469b0a(%rip), %eax   # 0x514e110 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451+0x8>
 3ce4606:      	testl	%eax, %eax
 3ce4608:      	jne	0x3ce47bb <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x85b>
 3ce460e:      	movq	0x1469af3(%rip), %rcx   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451>
 3ce4615:      	movq	%rbx, %rax
 3ce4618:      	orq	%r13, %rax
 3ce461b:      	shrq	$0x20, %rax
 3ce461f:      	je	0x3ce4720 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x7c0>
 3ce4625:      	movq	%rbx, %rax
 3ce4628:      	xorl	%edx, %edx
 3ce462a:      	divq	%r13
 3ce462d:      	movq	%rax, %rdx
 3ce4630:      	incq	%rdx
 3ce4633:      	movq	%rcx, %rax
 3ce4636:      	mulq	%rdx
 3ce4639:      	jo	0x3ce4738 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x7d8>
 3ce463f:      	movq	%rax, 0x60(%rsp)
 3ce4644:      	cmpq	0x28(%rsp), %rbx
 3ce4649:      	jae	0x3ce4a5b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xafb>
 3ce464f:      	callq	*0x145da8b(%rip)        # 0x51420e0 <writev+0x51420e0>
 3ce4655:      	movq	%r14, %rdi
 3ce4658:      	movq	%rax, %rsi
 3ce465b:      	callq	*0x145da8f(%rip)        # 0x51420f0 <writev+0x51420f0>
 3ce4661:      	cmpq	$0x0, 0x40(%rsp)
 3ce4667:      	sete	%cl
 3ce466a:      	imulq	$0x3e8, %rax, %rax      # imm = 0x3E8
 3ce4671:      	movl	%edx, %edx
 3ce4673:      	imulq	$0x431bde83, %rdx, %rsi # imm = 0x431BDE83
 3ce467a:      	shrq	$0x32, %rsi
 3ce467e:      	addq	%rax, %rsi
 3ce4681:      	testq	%rbx, %rbx
 3ce4684:      	setne	%al
 3ce4687:      	movq	%rsi, 0x78(%rsp)
 3ce468c:      	cmpb	$0x0, 0x24(%rsp)
 3ce4691:      	jne	0x3ce46d2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x772>
 3ce4693:      	orb	%al, %cl
 3ce4695:      	je	0x3ce46d2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x772>
 3ce4697:      	movq	0x60(%rsp), %rcx
 3ce469c:      	movq	%rcx, %rax
 3ce469f:      	movl	$0x3e8, %edx            # imm = 0x3E8
 3ce46a4:      	mulq	%rdx
 3ce46a7:      	jo	0x3ce47cc <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x86c>
 3ce46ad:      	cmpq	%rsi, %rax
 3ce46b0:      	ja	0x3ce4f14 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfb4>
 3ce46b6:      	movq	0xa8(%rsp), %rax
 3ce46be:      	testq	%rax, %rax
 3ce46c1:      	setne	%dl
 3ce46c4:      	cmpq	%rax, %rcx
 3ce46c7:      	seta	%al
 3ce46ca:      	testb	%al, %dl
 3ce46cc:      	jne	0x3ce5105 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x11a5>
 3ce46d2:      	leaq	0x1(%rbx), %rcx
 3ce46d6:      	movq	%rbx, %rax
 3ce46d9:      	movq	0xeb0(%rsp), %rdx
 3ce46e1:      	lock
 3ce46e2:      	cmpxchgq	%rcx, (%rdx,%r12,8)
 3ce46e7:      	je	0x3ce4c7d <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xd1d>
 3ce46ed:      	leaq	0xa30(%rsp), %rdi
 3ce46f5:      	callq	0x3c90bf0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEBF_>
 3ce46fa:      	xorl	%eax, %eax
 3ce46fc:      	movl	$0x1, %ecx
 3ce4701:      	lock
 3ce4702:      	cmpxchgl	%ecx, (%r15)
 3ce4706:      	je	0x3ce4440 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x4e0>
 3ce470c:      	jmp	0x3ce448b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x52b>
 3ce4711:      	nopw	%cs:(%rax,%rax)
 3ce4720:      	movl	%ebx, %eax
 3ce4722:      	xorl	%edx, %edx
 3ce4724:      	divl	%r13d
 3ce4727:      	movl	%eax, %edx
 3ce4729:      	incq	%rdx
 3ce472c:      	movq	%rcx, %rax
 3ce472f:      	mulq	%rdx
 3ce4732:      	jno	0x3ce463f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6df>
 3ce4738:      	movq	$-0x1, %rax
 3ce473f:      	movq	%rax, 0x60(%rsp)
 3ce4744:      	cmpq	0x28(%rsp), %rbx
 3ce4749:      	jb	0x3ce464f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6ef>
 3ce474f:      	jmp	0x3ce4a5b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xafb>
 3ce4754:      	nopw	%cs:(%rax,%rax)
 3ce4760:      	xorl	%ecx, %ecx
 3ce4762:      	ucomisd	-0x395bc1a(%rip), %xmm0 # 0x388b50 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.1382.llvm.6417388326552730234+0x28>
 3ce476a:      	movq	$-0x1, %rax
 3ce4771:      	ja	0x3ce4776 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x816>
 3ce4773:      	movq	%rcx, %rax
 3ce4776:      	movq	%rax, %rdx
 3ce4779:      	cmpq	$0x258, %rax            # imm = 0x258
 3ce477f:      	jae	0x3ce47a0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x840>
 3ce4781:      	movl	$0x1, %ecx
 3ce4786:      	testq	%rax, %rax
 3ce4789:      	je	0x3ce4615 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6b5>
 3ce478f:      	jmp	0x3ce47b3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x853>
 3ce4791:      	nopw	%cs:(%rax,%rax)
 3ce47a0:      	movl	$0x258, %edx            # imm = 0x258
 3ce47a5:      	movl	$0x1, %ecx
 3ce47aa:      	testq	%rax, %rax
 3ce47ad:      	je	0x3ce4615 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6b5>
 3ce47b3:      	movq	%rdx, %rcx
 3ce47b6:      	jmp	0x3ce4615 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6b5>
 3ce47bb:      	leaq	0x1469946(%rip), %rdi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451>
 3ce47c2:      	callq	0x3b2c922 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s0E0zEB1w_>
 3ce47c7:      	jmp	0x3ce460e <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6ae>
 3ce47cc:      	movq	$-0x1, %rax
 3ce47d3:      	cmpq	%rsi, %rax
 3ce47d6:      	jbe	0x3ce46b6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x756>
 3ce47dc:      	jmp	0x3ce4f14 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfb4>
 3ce47e1:      	movq	%r15, %rdi
 3ce47e4:      	callq	*0x145c886(%rip)        # 0x5141070 <writev+0x5141070>
 3ce47ea:      	jmp	0x3ce44f7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x597>
 3ce47ef:      	callq	*0x145c883(%rip)        # 0x5141078 <writev+0x5141078>
 3ce47f5:      	testb	%al, %al
 3ce47f7:      	jne	0x3ce44e6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x586>
 3ce47fd:      	movq	0x38(%rsp), %rax
 3ce4802:      	movb	$0x1, 0x14(%rax)
 3ce4806:      	jmp	0x3ce44e6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x586>
 3ce480b:      	movq	0x38(%rsp), %rdx
 3ce4810:      	shll	$0x3, %edx
 3ce4813:      	leaq	0x13772fe(%rip), %rax   # 0x505bb18 <anon.d42d8d1710be02f83480d8869995a290.3835.llvm.15708504647765068451+0xf0>
 3ce481a:      	movq	(%rdx,%rax), %rax
 3ce481e:      	leaq	-0x2eacae5(%rip), %rcx  # 0xe37d40 <anon.d42d8d1710be02f83480d8869995a290.3778.llvm.15708504647765068451+0x219>
 3ce4825:      	movq	(%rdx,%rcx), %rcx
 3ce4829:      	movq	%rax, 0xa30(%rsp)
 3ce4831:      	movq	%rcx, 0xa38(%rsp)
 3ce4839:      	movups	0x18(%rbp), %xmm0
 3ce483d:      	movups	%xmm0, 0x5f0(%rsp)
 3ce4845:      	leaq	0xa30(%rsp), %rax
 3ce484d:      	movq	%rax, 0xc0(%rsp)
 3ce4855:      	leaq	-0x2debdc(%rip), %rax   # 0x3a05c80 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtCs3pwlnhBXFtN_12memra_server>
 3ce485c:      	movq	%rax, 0xc8(%rsp)
 3ce4864:      	leaq	0x5f0(%rsp), %rax
 3ce486c:      	movq	%rax, 0xd0(%rsp)
 3ce4874:      	leaq	-0x2ded9b(%rip), %rax   # 0x3a05ae0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtReNtB6_5Debug3fmtCs3pwlnhBXFtN_12memra_server>
 3ce487b:      	movq	%rax, 0xd8(%rsp)
 3ce4883:      	leaq	0x50(%rsp), %rax
 3ce4888:      	movq	%rax, 0xe0(%rsp)
 3ce4890:      	movq	0x145c4d1(%rip), %rax   # 0x5140d68 <writev+0x5140d68>
 3ce4897:      	movq	%rax, 0xe8(%rsp)
 3ce489f:      	leaq	0x48(%rsp), %rcx
 3ce48a4:      	movq	%rcx, 0xf0(%rsp)
 3ce48ac:      	movq	%rax, 0xf8(%rsp)
 3ce48b4:      	leaq	0x28(%rsp), %rax
 3ce48b9:      	movq	%rax, 0x100(%rsp)
 3ce48c1:      	movq	0x145c5f8(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce48c8:      	movq	%rax, 0x108(%rsp)
 3ce48d0:      	leaq	-0x390276c(%rip), %rsi  # 0x3e216b <anon.fa2e791037615afd04c732fea021b56d.22.llvm.7881675451276153+0xc4>
 3ce48d7:      	leaq	0x78(%rsp), %rdi
 3ce48dc:      	leaq	0xc0(%rsp), %rdx
 3ce48e4:      	callq	*0x145c46e(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce48ea:      	movq	0x28(%rsp), %rbx
 3ce48ef:      	movq	0x80(%rsp), %r14
 3ce48f7:      	movq	0x88(%rsp), %rdx
 3ce48ff:      	leaq	-0x2ec3a4b(%rip), %r15  # 0xe20ebb <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x210>
 3ce4906:      	movq	%r15, 0x8(%rsp)
 3ce490b:      	movq	$0xa, 0x10(%rsp)
 3ce4914:      	leaq	-0x395dadb(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.15708504647765068451>
 3ce491b:      	leaq	0xc0(%rsp), %rdi
 3ce4923:      	movl	$0x10, %r8d
 3ce4929:      	movq	%r14, %rsi
 3ce492c:      	xorl	%r9d, %r9d
 3ce492f:      	callq	0x3cbb480 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4934:      	movq	0x100(%rsp), %rax
 3ce493c:      	movq	%rax, 0xa70(%rsp)
 3ce4944:      	movups	0xc0(%rsp), %xmm0
 3ce494c:      	movups	0xd0(%rsp), %xmm1
 3ce4954:      	movups	0xe0(%rsp), %xmm2
 3ce495c:      	movups	0xf0(%rsp), %xmm3
 3ce4964:      	movaps	%xmm3, 0xa60(%rsp)
 3ce496c:      	movaps	%xmm2, 0xa50(%rsp)
 3ce4974:      	movaps	%xmm1, 0xa40(%rsp)
 3ce497c:      	movaps	%xmm0, 0xa30(%rsp)
 3ce4984:      	leaq	0xc0(%rsp), %rdi
 3ce498c:      	leaq	0xa30(%rsp), %rsi
 3ce4994:      	callq	0x3eb0090 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce4999:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce49a3:      	leaq	0x5f0(%rsp), %rdi
 3ce49ab:      	leaq	0xc0(%rsp), %rsi
 3ce49b3:      	movl	$0x1, %edx
 3ce49b8:      	movq	%rbx, %rcx
 3ce49bb:      	callq	0x3ce1c70 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce49c0:      	movups	0x660(%rsp), %xmm0
 3ce49c8:      	movq	0x30(%rsp), %r12
 3ce49cd:      	movups	%xmm0, 0x70(%r12)
 3ce49d3:      	movups	0x650(%rsp), %xmm0
 3ce49db:      	movups	%xmm0, 0x60(%r12)
 3ce49e1:      	movups	0x640(%rsp), %xmm0
 3ce49e9:      	movups	%xmm0, 0x50(%r12)
 3ce49ef:      	movups	0x630(%rsp), %xmm0
 3ce49f7:      	movups	%xmm0, 0x40(%r12)
 3ce49fd:      	movups	0x5f0(%rsp), %xmm0
 3ce4a05:      	movups	0x600(%rsp), %xmm1
 3ce4a0d:      	movups	0x610(%rsp), %xmm2
 3ce4a15:      	movups	0x620(%rsp), %xmm3
 3ce4a1d:      	movups	%xmm3, 0x30(%r12)
 3ce4a23:      	movups	%xmm2, 0x20(%r12)
 3ce4a29:      	movups	%xmm1, 0x10(%r12)
 3ce4a2f:      	movups	%xmm0, (%r12)
 3ce4a34:      	movq	%r15, 0x80(%r12)
 3ce4a3c:      	movq	$0xa, 0x88(%r12)
 3ce4a48:      	movq	0x78(%rsp), %rsi
 3ce4a4d:      	testq	%rsi, %rsi
 3ce4a50:      	jne	0x3ce4f01 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfa1>
 3ce4a56:      	jmp	0x3ce50f0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4a5b:      	shll	$0x3, %r12d
 3ce4a5f:      	leaq	0x13770b2(%rip), %rax   # 0x505bb18 <anon.d42d8d1710be02f83480d8869995a290.3835.llvm.15708504647765068451+0xf0>
 3ce4a66:      	movq	(%r12,%rax), %rax
 3ce4a6a:      	leaq	-0x2eacd31(%rip), %rcx  # 0xe37d40 <anon.d42d8d1710be02f83480d8869995a290.3778.llvm.15708504647765068451+0x219>
 3ce4a71:      	movq	(%r12,%rcx), %rcx
 3ce4a75:      	movq	%rax, 0x78(%rsp)
 3ce4a7a:      	movq	%rcx, 0x80(%rsp)
 3ce4a82:      	leaq	0x78(%rsp), %rax
 3ce4a87:      	movq	%rax, 0xc0(%rsp)
 3ce4a8f:      	leaq	-0x2dee16(%rip), %rax   # 0x3a05c80 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtCs3pwlnhBXFtN_12memra_server>
 3ce4a96:      	movq	%rax, 0xc8(%rsp)
 3ce4a9e:      	leaq	0x58(%rsp), %rax
 3ce4aa3:      	movq	%rax, 0xd0(%rsp)
 3ce4aab:      	movq	0x145c2b6(%rip), %rax   # 0x5140d68 <writev+0x5140d68>
 3ce4ab2:      	movq	%rax, 0xd8(%rsp)
 3ce4aba:      	leaq	0x28(%rsp), %rcx
 3ce4abf:      	movq	%rcx, 0xe0(%rsp)
 3ce4ac7:      	movq	%rax, 0xe8(%rsp)
 3ce4acf:      	leaq	0x60(%rsp), %rax
 3ce4ad4:      	movq	%rax, 0xf0(%rsp)
 3ce4adc:      	movq	0x145c3dd(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce4ae3:      	movq	%rax, 0xf8(%rsp)
 3ce4aeb:      	leaq	-0x391c5ee(%rip), %rsi  # 0x3c8504 <anon.8d7079b8273dd0721474669710734b31.74.llvm.16532073193574447039+0x101>
 3ce4af2:      	leaq	0x5f0(%rsp), %rdi
 3ce4afa:      	leaq	0xc0(%rsp), %rdx
 3ce4b02:      	callq	*0x145c250(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4b08:      	movq	0x5f0(%rsp), %r14
 3ce4b10:      	movq	0x5f8(%rsp), %r15
 3ce4b18:      	movq	0x600(%rsp), %rdx
 3ce4b20:      	leaq	-0x2ec3c6c(%rip), %rbx  # 0xe20ebb <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x210>
 3ce4b27:      	movq	%rbx, 0x8(%rsp)
 3ce4b2c:      	movq	$0xa, 0x10(%rsp)
 3ce4b35:      	leaq	-0x395dcfc(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.15708504647765068451>
 3ce4b3c:      	leaq	0xc0(%rsp), %rdi
 3ce4b44:      	movl	$0x10, %r8d
 3ce4b4a:      	movq	%r15, %rsi
 3ce4b4d:      	xorl	%r9d, %r9d
 3ce4b50:      	callq	0x3cbb480 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4b55:      	movq	0x30(%rsp), %r12
 3ce4b5a:      	movq	0x100(%rsp), %rax
 3ce4b62:      	movq	%rax, 0x540(%rsp)
 3ce4b6a:      	movups	0xc0(%rsp), %xmm0
 3ce4b72:      	movups	0xd0(%rsp), %xmm1
 3ce4b7a:      	movupd	0xe0(%rsp), %xmm2
 3ce4b83:      	movupd	0xf0(%rsp), %xmm3
 3ce4b8c:      	movapd	%xmm3, 0x530(%rsp)
 3ce4b95:      	movapd	%xmm2, 0x520(%rsp)
 3ce4b9e:      	movaps	%xmm1, 0x510(%rsp)
 3ce4ba6:      	movaps	%xmm0, 0x500(%rsp)
 3ce4bae:      	leaq	0xc0(%rsp), %rdi
 3ce4bb6:      	leaq	0x500(%rsp), %rsi
 3ce4bbe:      	callq	0x3eb0090 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce4bc3:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce4bcd:      	movq	0x60(%rsp), %rcx
 3ce4bd2:      	leaq	0x5f0(%rsp), %rdi
 3ce4bda:      	leaq	0xc0(%rsp), %rsi
 3ce4be2:      	movl	$0x1, %edx
 3ce4be7:      	callq	0x3ce1c70 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce4bec:      	movups	0x660(%rsp), %xmm0
 3ce4bf4:      	movups	%xmm0, 0x70(%r12)
 3ce4bfa:      	movups	0x650(%rsp), %xmm0
 3ce4c02:      	movups	%xmm0, 0x60(%r12)
 3ce4c08:      	movups	0x640(%rsp), %xmm0
 3ce4c10:      	movups	%xmm0, 0x50(%r12)
 3ce4c16:      	movups	0x630(%rsp), %xmm0
 3ce4c1e:      	movups	%xmm0, 0x40(%r12)
 3ce4c24:      	movups	0x5f0(%rsp), %xmm0
 3ce4c2c:      	movups	0x600(%rsp), %xmm1
 3ce4c34:      	movups	0x610(%rsp), %xmm2
 3ce4c3c:      	movups	0x620(%rsp), %xmm3
 3ce4c44:      	movups	%xmm3, 0x30(%r12)
 3ce4c4a:      	movups	%xmm2, 0x20(%r12)
 3ce4c50:      	movups	%xmm1, 0x10(%r12)
 3ce4c56:      	movups	%xmm0, (%r12)
 3ce4c5b:      	movq	%rbx, 0x80(%r12)
 3ce4c63:      	movq	$0xa, 0x88(%r12)
 3ce4c6f:      	testq	%r14, %r14
 3ce4c72:      	jne	0x3ce50d2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1172>
 3ce4c78:      	jmp	0x3ce50e3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce4c7d:      	movq	0xeb8(%rsp), %rax
 3ce4c85:      	lock
 3ce4c86:      	incq	(%rax)
 3ce4c89:      	movq	0x30(%rsp), %r12
 3ce4c8e:      	movq	0xeb0(%rsp), %rcx
 3ce4c96:      	movq	%rcx, 0x8(%r12)
 3ce4c9b:      	movq	%rax, 0x10(%r12)
 3ce4ca0:      	movl	$0xffffffff, 0x30(%r12) # imm = 0xFFFFFFFF
 3ce4ca9:      	movl	0x24(%rsp), %eax
 3ce4cad:      	movb	%al, 0x38(%r12)
 3ce4cb2:      	movw	$0x1, 0x39(%r12)
 3ce4cba:      	movq	$-0x1, (%r12)
 3ce4cc2:      	jmp	0x3ce50e3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce4cc7:      	movq	0xeb8(%rsp), %rcx
 3ce4ccf:      	lock
 3ce4cd0:      	incq	(%rcx)
 3ce4cd3:      	movq	0x30(%rsp), %r12
 3ce4cd8:      	movq	0xeb0(%rsp), %rsi
 3ce4ce0:      	movq	%rsi, 0x8(%r12)
 3ce4ce5:      	movq	%rcx, 0x10(%r12)
 3ce4cea:      	movq	%rbx, 0x18(%r12)
 3ce4cef:      	movq	%rbp, 0x20(%r12)
 3ce4cf4:      	movq	%rax, 0x28(%r12)
 3ce4cf9:      	movl	%edx, 0x30(%r12)
 3ce4cfe:      	movl	0x24(%rsp), %eax
 3ce4d02:      	movb	%al, 0x38(%r12)
 3ce4d07:      	movw	$0x101, 0x39(%r12)      # imm = 0x101
 3ce4d0f:      	movq	$-0x1, (%r12)
 3ce4d17:      	jmp	0x3ce50f0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4d1c:      	movups	0x18(%rbp), %xmm0
 3ce4d20:      	movups	%xmm0, 0x5f0(%rsp)
 3ce4d28:      	leaq	0x28(%rsp), %rax
 3ce4d2d:      	movq	%rax, 0xc0(%rsp)
 3ce4d35:      	movq	0x145c184(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce4d3c:      	movq	%rax, 0xc8(%rsp)
 3ce4d44:      	leaq	0x5f0(%rsp), %rcx
 3ce4d4c:      	movq	%rcx, 0xd0(%rsp)
 3ce4d54:      	leaq	-0x2df27b(%rip), %rcx   # 0x3a05ae0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtReNtB6_5Debug3fmtCs3pwlnhBXFtN_12memra_server>
 3ce4d5b:      	movq	%rcx, 0xd8(%rsp)
 3ce4d63:      	leaq	0xb0(%rsp), %rcx
 3ce4d6b:      	movq	%rcx, 0xe0(%rsp)
 3ce4d73:      	movq	%rax, 0xe8(%rsp)
 3ce4d7b:      	leaq	-0x2ec4085(%rip), %rsi  # 0xe20cfd <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x52>
 3ce4d82:      	leaq	0x78(%rsp), %rdi
 3ce4d87:      	leaq	0xc0(%rsp), %rdx
 3ce4d8f:      	callq	*0x145bfc3(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4d95:      	movq	0x28(%rsp), %rbx
 3ce4d9a:      	movq	0x80(%rsp), %r14
 3ce4da2:      	movq	0x88(%rsp), %rdx
 3ce4daa:      	leaq	-0x2ec3ff2(%rip), %r15  # 0xe20dbf <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x114>
 3ce4db1:      	movq	%r15, 0x8(%rsp)
 3ce4db6:      	movq	$0xf, 0x10(%rsp)
 3ce4dbf:      	leaq	-0x395df86(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.15708504647765068451>
 3ce4dc6:      	leaq	0xc0(%rsp), %rdi
 3ce4dce:      	movl	$0x10, %r8d
 3ce4dd4:      	movq	%r14, %rsi
 3ce4dd7:      	xorl	%r9d, %r9d
 3ce4dda:      	callq	0x3cbb480 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4ddf:      	movq	0x100(%rsp), %rax
 3ce4de7:      	movq	%rax, 0xa70(%rsp)
 3ce4def:      	movups	0xc0(%rsp), %xmm0
 3ce4df7:      	movups	0xd0(%rsp), %xmm1
 3ce4dff:      	movups	0xe0(%rsp), %xmm2
 3ce4e07:      	movups	0xf0(%rsp), %xmm3
 3ce4e0f:      	movaps	%xmm3, 0xa60(%rsp)
 3ce4e17:      	movaps	%xmm2, 0xa50(%rsp)
 3ce4e1f:      	movaps	%xmm1, 0xa40(%rsp)
 3ce4e27:      	movaps	%xmm0, 0xa30(%rsp)
 3ce4e2f:      	leaq	0xc0(%rsp), %rdi
 3ce4e37:      	leaq	0xa30(%rsp), %rsi
 3ce4e3f:      	callq	0x3eb0090 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce4e44:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce4e4e:      	leaq	0x5f0(%rsp), %rdi
 3ce4e56:      	leaq	0xc0(%rsp), %rsi
 3ce4e5e:      	movl	$0x1, %edx
 3ce4e63:      	movq	%rbx, %rcx
 3ce4e66:      	callq	0x3ce1c70 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce4e6b:      	movups	0x660(%rsp), %xmm0
 3ce4e73:      	movq	0x30(%rsp), %r12
 3ce4e78:      	movups	%xmm0, 0x70(%r12)
 3ce4e7e:      	movups	0x650(%rsp), %xmm0
 3ce4e86:      	movups	%xmm0, 0x60(%r12)
 3ce4e8c:      	movups	0x640(%rsp), %xmm0
 3ce4e94:      	movups	%xmm0, 0x50(%r12)
 3ce4e9a:      	movups	0x630(%rsp), %xmm0
 3ce4ea2:      	movups	%xmm0, 0x40(%r12)
 3ce4ea8:      	movups	0x5f0(%rsp), %xmm0
 3ce4eb0:      	movups	0x600(%rsp), %xmm1
 3ce4eb8:      	movups	0x610(%rsp), %xmm2
 3ce4ec0:      	movups	0x620(%rsp), %xmm3
 3ce4ec8:      	movups	%xmm3, 0x30(%r12)
 3ce4ece:      	movups	%xmm2, 0x20(%r12)
 3ce4ed4:      	movups	%xmm1, 0x10(%r12)
 3ce4eda:      	movups	%xmm0, (%r12)
 3ce4edf:      	movq	%r15, 0x80(%r12)
 3ce4ee7:      	movq	$0xf, 0x88(%r12)
 3ce4ef3:      	movq	0x78(%rsp), %rsi
 3ce4ef8:      	testq	%rsi, %rsi
 3ce4efb:      	je	0x3ce50f0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4f01:      	movl	$0x1, %edx
 3ce4f06:      	movq	%r14, %rdi
 3ce4f09:      	callq	*0x145be31(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce4f0f:      	jmp	0x3ce50f0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4f14:      	leaq	0x60(%rsp), %rax
 3ce4f19:      	movq	%rax, 0xc0(%rsp)
 3ce4f21:      	movq	0x145bf98(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce4f28:      	movq	%rax, 0xc8(%rsp)
 3ce4f30:      	leaq	0x78(%rsp), %rcx
 3ce4f35:      	movq	%rcx, 0xd0(%rsp)
 3ce4f3d:      	movq	%rax, 0xd8(%rsp)
 3ce4f45:      	leaq	-0x2ec329e(%rip), %rsi  # 0xe21cae <anon.d42d8d1710be02f83480d8869995a290.1045.llvm.15708504647765068451+0x89>
 3ce4f4c:      	leaq	0x5f0(%rsp), %rdi
 3ce4f54:      	leaq	0xc0(%rsp), %rdx
 3ce4f5c:      	callq	*0x145bdf6(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4f62:      	movq	0x5f0(%rsp), %r14
 3ce4f6a:      	movq	0x5f8(%rsp), %r15
 3ce4f72:      	movq	0x600(%rsp), %rdx
 3ce4f7a:      	leaq	-0x2ec40d3(%rip), %rbx  # 0xe20eae <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x203>
 3ce4f81:      	movq	%rbx, 0x8(%rsp)
 3ce4f86:      	movq	$0xd, 0x10(%rsp)
 3ce4f8f:      	leaq	-0x395e156(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.15708504647765068451>
 3ce4f96:      	leaq	0xc0(%rsp), %rdi
 3ce4f9e:      	movl	$0x10, %r8d
 3ce4fa4:      	movq	%r15, %rsi
 3ce4fa7:      	xorl	%r9d, %r9d
 3ce4faa:      	callq	0x3cbb480 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4faf:      	movq	0x30(%rsp), %r12
 3ce4fb4:      	movq	0x100(%rsp), %rax
 3ce4fbc:      	movq	%rax, 0x590(%rsp)
 3ce4fc4:      	movups	0xc0(%rsp), %xmm0
 3ce4fcc:      	movups	0xd0(%rsp), %xmm1
 3ce4fd4:      	movupd	0xe0(%rsp), %xmm2
 3ce4fdd:      	movupd	0xf0(%rsp), %xmm3
 3ce4fe6:      	movapd	%xmm3, 0x580(%rsp)
 3ce4fef:      	movapd	%xmm2, 0x570(%rsp)
 3ce4ff8:      	movaps	%xmm1, 0x560(%rsp)
 3ce5000:      	movaps	%xmm0, 0x550(%rsp)
 3ce5008:      	leaq	0xc0(%rsp), %rdi
 3ce5010:      	leaq	0x550(%rsp), %rsi
 3ce5018:      	callq	0x3eb0090 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce501d:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce5027:      	movq	0x60(%rsp), %rcx
 3ce502c:      	leaq	0x5f0(%rsp), %rdi
 3ce5034:      	leaq	0xc0(%rsp), %rsi
 3ce503c:      	movl	$0x1, %edx
 3ce5041:      	callq	0x3ce1c70 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce5046:      	movups	0x660(%rsp), %xmm0
 3ce504e:      	movups	%xmm0, 0x70(%r12)
 3ce5054:      	movups	0x650(%rsp), %xmm0
 3ce505c:      	movups	%xmm0, 0x60(%r12)
 3ce5062:      	movups	0x640(%rsp), %xmm0
 3ce506a:      	movups	%xmm0, 0x50(%r12)
 3ce5070:      	movups	0x630(%rsp), %xmm0
 3ce5078:      	movups	%xmm0, 0x40(%r12)
 3ce507e:      	movups	0x5f0(%rsp), %xmm0
 3ce5086:      	movups	0x600(%rsp), %xmm1
 3ce508e:      	movupd	0x610(%rsp), %xmm2
 3ce5097:      	movupd	0x620(%rsp), %xmm3
 3ce50a0:      	movupd	%xmm3, 0x30(%r12)
 3ce50a7:      	movupd	%xmm2, 0x20(%r12)
 3ce50ae:      	movups	%xmm1, 0x10(%r12)
 3ce50b4:      	movups	%xmm0, (%r12)
 3ce50b9:      	movq	%rbx, 0x80(%r12)
 3ce50c1:      	movq	$0xd, 0x88(%r12)
 3ce50cd:      	testq	%r14, %r14
 3ce50d0:      	je	0x3ce50e3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce50d2:      	movl	$0x1, %edx
 3ce50d7:      	movq	%r15, %rdi
 3ce50da:      	movq	%r14, %rsi
 3ce50dd:      	callq	*0x145bc5d(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce50e3:      	leaq	0xa30(%rsp), %rdi
 3ce50eb:      	callq	0x3c90bf0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEBF_>
 3ce50f0:      	movq	%r12, %rax
 3ce50f3:      	addq	$0xe78, %rsp            # imm = 0xE78
 3ce50fa:      	popq	%rbx
 3ce50fb:      	popq	%r12
 3ce50fd:      	popq	%r13
 3ce50ff:      	popq	%r14
 3ce5101:      	popq	%r15
 3ce5103:      	popq	%rbp
 3ce5104:      	retq
 3ce5105:      	leaq	0x60(%rsp), %rax
 3ce510a:      	movq	%rax, 0xc0(%rsp)
 3ce5112:      	movq	0x145bda7(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce5119:      	movq	%rax, 0xc8(%rsp)
 3ce5121:      	leaq	0xa8(%rsp), %rcx
 3ce5129:      	movq	%rcx, 0xd0(%rsp)
 3ce5131:      	movq	%rax, 0xd8(%rsp)
 3ce5139:      	leaq	-0x2ec33be(%rip), %rsi  # 0xe21d82 <anon.d42d8d1710be02f83480d8869995a290.1045.llvm.15708504647765068451+0x15d>
 3ce5140:      	leaq	0x5f0(%rsp), %rdi
 3ce5148:      	leaq	0xc0(%rsp), %rdx
 3ce5150:      	callq	*0x145bc02(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce5156:      	movq	0x5f0(%rsp), %r14
 3ce515e:      	movq	0x5f8(%rsp), %r15
 3ce5166:      	movq	0x600(%rsp), %rdx
 3ce516e:      	leaq	-0x2ec43b6(%rip), %rbx  # 0xe20dbf <anon.d42d8d1710be02f83480d8869995a290.867.llvm.15708504647765068451+0x114>
 3ce5175:      	movq	%rbx, 0x8(%rsp)
 3ce517a:      	movq	$0xf, 0x10(%rsp)
 3ce5183:      	leaq	-0x395e34a(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.15708504647765068451>
 3ce518a:      	leaq	0xc0(%rsp), %rdi
 3ce5192:      	movl	$0x10, %r8d
 3ce5198:      	movq	%r15, %rsi
 3ce519b:      	xorl	%r9d, %r9d
 3ce519e:      	callq	0x3cbb480 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce51a3:      	movq	0x30(%rsp), %r12
 3ce51a8:      	movq	0x100(%rsp), %rax
 3ce51b0:      	movq	%rax, 0x5e0(%rsp)
 3ce51b8:      	movups	0xc0(%rsp), %xmm0
 3ce51c0:      	movups	0xd0(%rsp), %xmm1
 3ce51c8:      	movupd	0xe0(%rsp), %xmm2
 3ce51d1:      	movupd	0xf0(%rsp), %xmm3
 3ce51da:      	movapd	%xmm3, 0x5d0(%rsp)
 3ce51e3:      	movapd	%xmm2, 0x5c0(%rsp)
 3ce51ec:      	movaps	%xmm1, 0x5b0(%rsp)
 3ce51f4:      	movaps	%xmm0, 0x5a0(%rsp)
 3ce51fc:      	leaq	0xc0(%rsp), %rdi
 3ce5204:      	leaq	0x5a0(%rsp), %rsi
 3ce520c:      	callq	0x3eb0090 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce5211:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce521b:      	movq	0x60(%rsp), %rcx
 3ce5220:      	leaq	0x5f0(%rsp), %rdi
 3ce5228:      	leaq	0xc0(%rsp), %rsi
 3ce5230:      	movl	$0x1, %edx
 3ce5235:      	callq	0x3ce1c70 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce523a:      	movups	0x660(%rsp), %xmm0
 3ce5242:      	movups	%xmm0, 0x70(%r12)
 3ce5248:      	movups	0x650(%rsp), %xmm0
 3ce5250:      	movups	%xmm0, 0x60(%r12)
 3ce5256:      	movups	0x640(%rsp), %xmm0
 3ce525e:      	movups	%xmm0, 0x50(%r12)
 3ce5264:      	movups	0x630(%rsp), %xmm0
 3ce526c:      	movups	%xmm0, 0x40(%r12)
 3ce5272:      	movups	0x5f0(%rsp), %xmm0
 3ce527a:      	movups	0x600(%rsp), %xmm1
 3ce5282:      	movupd	0x610(%rsp), %xmm2
 3ce528b:      	movupd	0x620(%rsp), %xmm3
 3ce5294:      	movupd	%xmm3, 0x30(%r12)
 3ce529b:      	movupd	%xmm2, 0x20(%r12)
 3ce52a2:      	movups	%xmm1, 0x10(%r12)
 3ce52a8:      	movups	%xmm0, (%r12)
 3ce52ad:      	movq	%rbx, 0x80(%r12)
 3ce52b5:      	movq	$0xf, 0x88(%r12)
 3ce52c1:      	testq	%r14, %r14
 3ce52c4:      	jne	0x3ce50d2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1172>
 3ce52ca:      	jmp	0x3ce50e3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce52cf:      	leaq	0x1468dfa(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 3ce52d6:      	callq	0x3b2b3a1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 3ce52db:      	jmp	0x3ce3fbe <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x5e>
 3ce52e0:      	testb	%cl, %cl
 3ce52e2:      	movq	$-0x1, %rcx
 3ce52e9:      	cmovneq	%rax, %rcx
 3ce52ed:      	movq	%rcx, 0x48(%rsp)
 3ce52f2:      	jmp	0x3ce4003 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xa3>
 3ce52f7:      	leaq	0x1468e72(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.15708504647765068451>
 3ce52fe:      	movq	%rcx, %rbx
 3ce5301:      	movq	%rsi, %r15
 3ce5304:      	callq	0x3b2b248 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 3ce5309:      	movq	%r15, %rsi
 3ce530c:      	movq	%rbx, %rcx
 3ce530f:      	jmp	0x3ce43a0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x440>
 3ce5314:      	leaq	0x1468db5(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 3ce531b:      	movq	%rcx, 0x38(%rsp)
 3ce5320:      	movq	%rsi, %r15
 3ce5323:      	callq	0x3b2b3a1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 3ce5328:      	movq	%r15, %rsi
 3ce532b:      	movq	0x38(%rsp), %rcx
 3ce5330:      	movq	0x1468d99(%rip), %rdx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451>
 3ce5337:      	movq	0x1468d9a(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.15708504647765068451+0x8>
 3ce533e:      	shrq	$0x3e, %rbx
 3ce5342:      	je	0x3ce43e2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x482>
 3ce5348:      	movq	$-0x1, %rdi
 3ce534f:      	jmp	0x3ce43ea <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x48a>
 3ce5354:      	ud2
 3ce5356:      	movq	0x30(%rbp,%rbx,8), %rax
 3ce535b:      	movq	%rax, 0x50(%rsp)
 3ce5360:      	movq	0x30(%rbp), %rax
 3ce5364:      	movq	0x38(%rbp), %rax
 3ce5368:      	movq	0x40(%rbp), %rax
 3ce536c:      	movl	0x1468d9e(%rip), %eax   # 0x514e110 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451+0x8>
 3ce5372:      	testl	%eax, %eax
 3ce5374:      	jne	0x3ce5394 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1434>
 3ce5376:      	movq	0x1468d8b(%rip), %rsi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451>
 3ce537d:      	movq	0x40(%rsp), %rdi
 3ce5382:      	callq	0x39b5bb0 <_RNvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB4_9RouteLoad18service_estimate_s>
 3ce5387:      	leaq	0x13489ea(%rip), %rdi   # 0x502dd78 <anon.d42d8d1710be02f83480d8869995a290.849.llvm.15708504647765068451+0x58>
 3ce538e:      	callq	*0x145b9e4(%rip)        # 0x5140d78 <writev+0x5140d78>
 3ce5394:      	leaq	0x1468d6d(%rip), %rdi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.15708504647765068451>
 3ce539b:      	callq	0x3b2c922 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s0E0zEB1w_>
 3ce53a0:      	jmp	0x3ce5376 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1416>
 3ce53a2:      	jmp	0x3ce53ac <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x144c>
 3ce53a4:      	jmp	0x3ce53ac <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x144c>
 3ce53a6:      	jmp	0x3ce541f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14bf>
 3ce53a8:      	jmp	0x3ce53c7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1467>
 3ce53aa:      	jmp	0x3ce53c7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1467>
 3ce53ac:      	movq	%rax, %r12
 3ce53af:      	testq	%r14, %r14
 3ce53b2:      	je	0x3ce5422 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14c2>
 3ce53b4:      	movl	$0x1, %edx
 3ce53b9:      	movq	%r15, %rdi
 3ce53bc:      	movq	%r14, %rsi
 3ce53bf:      	callq	*0x145b97b(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce53c5:      	jmp	0x3ce5422 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14c2>
 3ce53c7:      	movq	%rax, %r12
 3ce53ca:      	movq	0x78(%rsp), %rsi
 3ce53cf:      	testq	%rsi, %rsi
 3ce53d2:      	je	0x3ce542f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce53d4:      	movl	$0x1, %edx
 3ce53d9:      	movq	%r14, %rdi
 3ce53dc:      	callq	*0x145b95e(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce53e2:      	movq	%r12, %rdi
 3ce53e5:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3ce53ea:      	movq	%rax, %r12
 3ce53ed:      	movzbl	%bl, %esi
 3ce53f0:      	movq	%r15, %rdi
 3ce53f3:      	callq	0x3c89c70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardINtNtB4_6option6OptionNtNtNtCs8e8YMT7Sgk_5tokio4sync9semaphore20OwnedSemaphorePermitEEECs3pwlnhBXFtN_12memra_server>
 3ce53f8:      	jmp	0x3ce542f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce53fa:      	callq	*0x145b9b8(%rip)        # 0x5140db8 <writev+0x5140db8>
 3ce5400:      	movq	%rax, %r12
 3ce5403:      	lock
 3ce5404:      	decq	(%rbp)
 3ce5408:      	jne	0x3ce542f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce540a:      	leaq	0xc0(%rsp), %rdi
 3ce5412:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3ce5417:      	jmp	0x3ce542f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce5419:      	callq	*0x145b999(%rip)        # 0x5140db8 <writev+0x5140db8>
 3ce541f:      	movq	%rax, %r12
 3ce5422:      	leaq	0xa30(%rsp), %rdi
 3ce542a:      	callq	0x3c90bf0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEBF_>
 3ce542f:      	movq	%r12, %rdi
 3ce5432:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3ce5437:      	callq	*0x145b97b(%rip)        # 0x5140db8 <writev+0x5140db8>
 3ce543d:      	int3
 3ce543e:      	int3
 3ce543f:      	int3

0000000003d869c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 3d869c0:      	pushq	%r14
 3d869c2:      	pushq	%rbx
 3d869c3:      	pushq	%rax
 3d869c4:      	movq	%rdi, %rbx
 3d869c7:      	cmpb	$0x1, 0x31(%rdi)
 3d869cb:      	jne	0x3d86a21 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3d869cd:      	movq	0x8(%rbx), %rcx
 3d869d1:      	movq	(%rcx), %rax
 3d869d4:      	nopw	%cs:(%rax,%rax)
 3d869e0:      	testq	%rax, %rax
 3d869e3:      	je	0x3d869f0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 3d869e5:      	leaq	-0x1(%rax), %rdx
 3d869e9:      	lock
 3d869ea:      	cmpxchgq	%rdx, (%rcx)
 3d869ee:      	jne	0x3d869e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 3d869f0:      	cmpb	$0x0, 0x32(%rbx)
 3d869f4:      	jne	0x3d86a21 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3d869f6:      	movq	(%rbx), %rcx
 3d869f9:      	movzbl	0x30(%rbx), %edx
 3d869fd:      	movq	(%rcx,%rdx,8), %rax
 3d86a01:      	nopw	%cs:(%rax,%rax)
 3d86a10:      	testq	%rax, %rax
 3d86a13:      	je	0x3d86a21 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3d86a15:      	leaq	-0x1(%rax), %rsi
 3d86a19:      	lock
 3d86a1a:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 3d86a1f:      	jne	0x3d86a10 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 3d86a21:      	cmpl	$-0x1, 0x28(%rbx)
 3d86a25:      	je	0x3d86a6a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3d86a27:      	movq	0x10(%rbx), %rdi
 3d86a2b:      	cmpq	$0x2, %rdi
 3d86a2f:      	ja	0x3d86a72 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 3d86a31:      	movq	0x18(%rbx), %rcx
 3d86a35:      	addq	$0x18, %rbx
 3d86a39:      	movq	0x30(%rcx,%rdi,8), %rax
 3d86a3e:      	nop
 3d86a40:      	testq	%rax, %rax
 3d86a43:      	je	0x3d86a52 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 3d86a45:      	leaq	-0x1(%rax), %rdx
 3d86a49:      	lock
 3d86a4a:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 3d86a50:      	jne	0x3d86a40 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 3d86a52:      	movq	(%rbx), %rax
 3d86a55:      	lock
 3d86a56:      	decq	(%rax)
 3d86a59:      	jne	0x3d86a6a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3d86a5b:      	movq	%rbx, %rdi
 3d86a5e:      	addq	$0x8, %rsp
 3d86a62:      	popq	%rbx
 3d86a63:      	popq	%r14
 3d86a65:      	jmp	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3d86a6a:      	addq	$0x8, %rsp
 3d86a6e:      	popq	%rbx
 3d86a6f:      	popq	%r14
 3d86a71:      	retq
 3d86a72:      	leaq	0x129b457(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 3d86a79:      	movl	$0x3, %esi
 3d86a7e:      	callq	*0x13ba3f4(%rip)        # 0x5140e78 <writev+0x5140e78>
 3d86a84:      	ud2
 3d86a86:      	movq	%rax, %r14
 3d86a89:      	movq	0x18(%rbx), %rax
 3d86a8d:      	lock
 3d86a8e:      	decq	(%rax)
 3d86a91:      	jne	0x3d86a9f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 3d86a93:      	addq	$0x18, %rbx
 3d86a97:      	movq	%rbx, %rdi
 3d86a9a:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3d86a9f:      	movq	%r14, %rdi
 3d86aa2:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3d86aa7:      	callq	*0x13ba30b(%rip)        # 0x5140db8 <writev+0x5140db8>
 3d86aad:      	int3
 3d86aae:      	int3
 3d86aaf:      	int3

0000000003e589a0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 3e589a0:      	pushq	%r14
 3e589a2:      	pushq	%rbx
 3e589a3:      	pushq	%rax
 3e589a4:      	movq	%rdi, %rbx
 3e589a7:      	cmpb	$0x1, 0x31(%rdi)
 3e589ab:      	jne	0x3e58a01 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3e589ad:      	movq	0x8(%rbx), %rcx
 3e589b1:      	movq	(%rcx), %rax
 3e589b4:      	nopw	%cs:(%rax,%rax)
 3e589c0:      	testq	%rax, %rax
 3e589c3:      	je	0x3e589d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 3e589c5:      	leaq	-0x1(%rax), %rdx
 3e589c9:      	lock
 3e589ca:      	cmpxchgq	%rdx, (%rcx)
 3e589ce:      	jne	0x3e589c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 3e589d0:      	cmpb	$0x0, 0x32(%rbx)
 3e589d4:      	jne	0x3e58a01 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3e589d6:      	movq	(%rbx), %rcx
 3e589d9:      	movzbl	0x30(%rbx), %edx
 3e589dd:      	movq	(%rcx,%rdx,8), %rax
 3e589e1:      	nopw	%cs:(%rax,%rax)
 3e589f0:      	testq	%rax, %rax
 3e589f3:      	je	0x3e58a01 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3e589f5:      	leaq	-0x1(%rax), %rsi
 3e589f9:      	lock
 3e589fa:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 3e589ff:      	jne	0x3e589f0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 3e58a01:      	cmpl	$-0x1, 0x28(%rbx)
 3e58a05:      	je	0x3e58a4a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3e58a07:      	movq	0x10(%rbx), %rdi
 3e58a0b:      	cmpq	$0x2, %rdi
 3e58a0f:      	ja	0x3e58a52 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 3e58a11:      	movq	0x18(%rbx), %rcx
 3e58a15:      	addq	$0x18, %rbx
 3e58a19:      	movq	0x30(%rcx,%rdi,8), %rax
 3e58a1e:      	nop
 3e58a20:      	testq	%rax, %rax
 3e58a23:      	je	0x3e58a32 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 3e58a25:      	leaq	-0x1(%rax), %rdx
 3e58a29:      	lock
 3e58a2a:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 3e58a30:      	jne	0x3e58a20 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 3e58a32:      	movq	(%rbx), %rax
 3e58a35:      	lock
 3e58a36:      	decq	(%rax)
 3e58a39:      	jne	0x3e58a4a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3e58a3b:      	movq	%rbx, %rdi
 3e58a3e:      	addq	$0x8, %rsp
 3e58a42:      	popq	%rbx
 3e58a43:      	popq	%r14
 3e58a45:      	jmp	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3e58a4a:      	addq	$0x8, %rsp
 3e58a4e:      	popq	%rbx
 3e58a4f:      	popq	%r14
 3e58a51:      	retq
 3e58a52:      	leaq	0x11c9477(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 3e58a59:      	movl	$0x3, %esi
 3e58a5e:      	callq	*0x12e8414(%rip)        # 0x5140e78 <writev+0x5140e78>
 3e58a64:      	ud2
 3e58a66:      	movq	%rax, %r14
 3e58a69:      	movq	0x18(%rbx), %rax
 3e58a6d:      	lock
 3e58a6e:      	decq	(%rax)
 3e58a71:      	jne	0x3e58a7f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 3e58a73:      	addq	$0x18, %rbx
 3e58a77:      	movq	%rbx, %rdi
 3e58a7a:      	callq	0x39bcc60 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3e58a7f:      	movq	%r14, %rdi
 3e58a82:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3e58a87:      	callq	*0x12e832b(%rip)        # 0x5140db8 <writev+0x5140db8>
 3e58a8d:      	int3
 3e58a8e:      	int3
 3e58a8f:      	int3

0000000003ebd040 <_RNvCs9wFQrvczXsK_7___rustc12___rust_alloc>:
 3ebd040:      	jmp	0x4f8f3f0 <_RNvCs9wFQrvczXsK_7___rustc11___rdl_alloc>
 3ebd045:      	int3
 3ebd046:      	int3
 3ebd047:      	int3
 3ebd048:      	int3
 3ebd049:      	int3
 3ebd04a:      	int3
 3ebd04b:      	int3
 3ebd04c:      	int3
 3ebd04d:      	int3
 3ebd04e:      	int3
 3ebd04f:      	int3

0000000003ebd050 <_RNvCs9wFQrvczXsK_7___rustc14___rust_dealloc>:
 3ebd050:      	jmp	0x4f8f450 <_RNvCs9wFQrvczXsK_7___rustc13___rdl_dealloc>
 3ebd055:      	int3
 3ebd056:      	int3
 3ebd057:      	int3
 3ebd058:      	int3
 3ebd059:      	int3
 3ebd05a:      	int3
 3ebd05b:      	int3
 3ebd05c:      	int3
 3ebd05d:      	int3
 3ebd05e:      	int3
 3ebd05f:      	int3

0000000003ebd080 <_RNvCs9wFQrvczXsK_7___rustc35___rust_no_alloc_shim_is_unstable_v2>:
 3ebd080:      	retq
 3ebd081:      	int3
 3ebd082:      	int3
 3ebd083:      	int3
 3ebd084:      	int3
 3ebd085:      	int3
 3ebd086:      	int3
 3ebd087:      	int3
 3ebd088:      	int3
 3ebd089:      	int3
 3ebd08a:      	int3
 3ebd08b:      	int3
 3ebd08c:      	int3
 3ebd08d:      	int3
 3ebd08e:      	int3
 3ebd08f:      	int3

0000000004bcaae0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArceE9drop_slowCs4Uug9gnAtr7_4axum>:
 4bcaae0:      	movq	%rdi, %rax
 4bcaae3:      	movq	(%rdi), %rdi
 4bcaae6:      	cmpq	$-0x1, %rdi
 4bcaaea:      	je	0x4bcab0c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArceE9drop_slowCs4Uug9gnAtr7_4axum+0x2c>
 4bcaaec:      	movq	0x8(%rax), %rsi
 4bcaaf0:      	lock
 4bcaaf1:      	decq	0x8(%rdi)
 4bcaaf5:      	jne	0x4bcab0c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArceE9drop_slowCs4Uug9gnAtr7_4axum+0x2c>
 4bcaaf7:      	addq	$0x17, %rsi
 4bcaafb:      	andq	$-0x8, %rsi
 4bcaaff:      	je	0x4bcab0c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArceE9drop_slowCs4Uug9gnAtr7_4axum+0x2c>
 4bcab01:      	movl	$0x8, %edx
 4bcab06:      	jmpq	*0x576234(%rip)         # 0x5140d40 <writev+0x5140d40>
 4bcab0c:      	retq
 4bcab0d:      	int3
 4bcab0e:      	int3
 4bcab0f:      	int3

0000000004c6ce70 <_RNvMNtNtCs8e8YMT7Sgk_5tokio4time7instantNtB2_7Instant25saturating_duration_since>:
 4c6ce70:      	jmpq	*0x4d4572(%rip)         # 0x51413e8 <writev+0x51413e8>
 4c6ce76:      	int3
 4c6ce77:      	int3
 4c6ce78:      	int3
 4c6ce79:      	int3
 4c6ce7a:      	int3
 4c6ce7b:      	int3
 4c6ce7c:      	int3
 4c6ce7d:      	int3
 4c6ce7e:      	int3
 4c6ce7f:      	int3

0000000004c6ce80 <_RNvMNtNtCs8e8YMT7Sgk_5tokio4time7instantNtB2_7Instant3now>:
 4c6ce80:      	jmp	0x4c73cb0 <_RNvNtNtCs8e8YMT7Sgk_5tokio4time5clock3now>
 4c6ce85:      	int3
 4c6ce86:      	int3
 4c6ce87:      	int3
 4c6ce88:      	int3
 4c6ce89:      	int3
 4c6ce8a:      	int3
 4c6ce8b:      	int3
 4c6ce8c:      	int3
 4c6ce8d:      	int3
 4c6ce8e:      	int3
 4c6ce8f:      	int3

0000000004c70080 <_RNvXs1_NtNtCs8e8YMT7Sgk_5tokio4time7instantNtB5_7InstantINtNtNtCs4NRVxsYgnAr_4core3ops5arith3AddNtNtBZ_4time8DurationE3add>:
 4c70080:      	leaq	0x4677a9(%rip), %r8     # 0x50d7830 <anon.997af0ac766256136b35917f9aa78705.74.llvm.7860900753440259477>
 4c70087:      	jmpq	*0x4d13c3(%rip)         # 0x5141450 <writev+0x5141450>
 4c7008d:      	int3
 4c7008e:      	int3
 4c7008f:      	int3

0000000004c7eda0 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env>:
 4c7eda0:      	pushq	%rbp
 4c7eda1:      	pushq	%r15
 4c7eda3:      	pushq	%r14
 4c7eda5:      	pushq	%r13
 4c7eda7:      	pushq	%r12
 4c7eda9:      	pushq	%rbx
 4c7edaa:      	subq	$0x38, %rsp
 4c7edae:      	movq	%rdi, %rbx
 4c7edb1:      	leaq	-0x48f6648(%rip), %rsi  # 0x388770 <anon.f57006984c741f17b95ade2a26912763.111.llvm.8110140741785617923+0x190>
 4c7edb8:      	leaq	0x8(%rsp), %rdi
 4c7edbd:      	movl	$0x10, %edx
 4c7edc2:      	callq	*0x4c2080(%rip)         # 0x5140e48 <writev+0x5140e48>
 4c7edc8:      	cmpl	$0x1, 0x8(%rsp)
 4c7edcd:      	jne	0x4c7edf0 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x50>
 4c7edcf:      	movq	0x10(%rsp), %rsi
 4c7edd4:      	testq	%rsi, %rsi
 4c7edd7:      	jle	0x4c7ede9 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x49>
 4c7edd9:      	movq	0x18(%rsp), %rdi
 4c7edde:      	movl	$0x1, %edx
 4c7ede3:      	callq	*0x4c1f57(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7ede9:      	movl	$0x42480000, %ebp       # imm = 0x42480000
 4c7edee:      	jmp	0x4c7ee39 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x99>
 4c7edf0:      	movq	0x10(%rsp), %r14
 4c7edf5:      	movl	$0x42480000, %ebp       # imm = 0x42480000
 4c7edfa:      	cmpq	$-0x1, %r14
 4c7edfe:      	je	0x4c7ee39 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x99>
 4c7ee00:      	movq	0x18(%rsp), %r15
 4c7ee05:      	movq	0x20(%rsp), %rsi
 4c7ee0a:      	movq	%r15, %rdi
 4c7ee0d:      	callq	*0x4c2505(%rip)         # 0x5141318 <writev+0x5141318>
 4c7ee13:      	movq	%rax, %r12
 4c7ee16:      	testq	%r14, %r14
 4c7ee19:      	je	0x4c7ee2c <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x8c>
 4c7ee1b:      	movl	$0x1, %edx
 4c7ee20:      	movq	%r15, %rdi
 4c7ee23:      	movq	%r14, %rsi
 4c7ee26:      	callq	*0x4c1f14(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7ee2c:      	testb	$0x1, %r12b
 4c7ee30:      	jne	0x4c7ee39 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x99>
 4c7ee32:      	shrq	$0x20, %r12
 4c7ee36:      	movl	%r12d, %ebp
 4c7ee39:      	leaq	-0x2514598(%rip), %rdi  # 0x276a8a8 <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x204>
 4c7ee40:      	movl	$0x16, %esi
 4c7ee45:      	movl	$0x20, %edx
 4c7ee4a:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7ee4f:      	movq	%rax, %r14
 4c7ee52:      	leaq	-0x48fa499(%rip), %rsi  # 0x3849c0 <anon.d42d8d1710be02f83480d8869995a290.115.llvm.15708504647765068451+0x310>
 4c7ee59:      	leaq	0x8(%rsp), %rdi
 4c7ee5e:      	movl	$0x10, %edx
 4c7ee63:      	callq	*0x4c1fdf(%rip)         # 0x5140e48 <writev+0x5140e48>
 4c7ee69:      	cmpl	$0x1, 0x8(%rsp)
 4c7ee6e:      	jne	0x4c7ee94 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0xf4>
 4c7ee70:      	movq	0x10(%rsp), %rsi
 4c7ee75:      	testq	%rsi, %rsi
 4c7ee78:      	jle	0x4c7ee8a <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0xea>
 4c7ee7a:      	movq	0x18(%rsp), %rdi
 4c7ee7f:      	movl	$0x1, %edx
 4c7ee84:      	callq	*0x4c1eb6(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7ee8a:      	movl	$0x3f800000, 0x4(%rsp)  # imm = 0x3F800000
 4c7ee92:      	jmp	0x4c7eee2 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x142>
 4c7ee94:      	movq	0x10(%rsp), %r15
 4c7ee99:      	movl	$0x3f800000, 0x4(%rsp)  # imm = 0x3F800000
 4c7eea1:      	cmpq	$-0x1, %r15
 4c7eea5:      	je	0x4c7eee2 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x142>
 4c7eea7:      	movq	0x18(%rsp), %r12
 4c7eeac:      	movq	0x20(%rsp), %rsi
 4c7eeb1:      	movq	%r12, %rdi
 4c7eeb4:      	callq	*0x4c245e(%rip)         # 0x5141318 <writev+0x5141318>
 4c7eeba:      	movq	%rax, %r13
 4c7eebd:      	testq	%r15, %r15
 4c7eec0:      	je	0x4c7eed3 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x133>
 4c7eec2:      	movl	$0x1, %edx
 4c7eec7:      	movq	%r12, %rdi
 4c7eeca:      	movq	%r15, %rsi
 4c7eecd:      	callq	*0x4c1e6d(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7eed3:      	testb	$0x1, %r13b
 4c7eed7:      	jne	0x4c7eee2 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x142>
 4c7eed9:      	shrq	$0x20, %r13
 4c7eedd:      	movl	%r13d, 0x4(%rsp)
 4c7eee2:      	leaq	-0x251462b(%rip), %rsi  # 0x276a8be <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x21a>
 4c7eee9:      	leaq	0x8(%rsp), %rdi
 4c7eeee:      	movl	$0x12, %edx
 4c7eef3:      	callq	*0x4c1f4f(%rip)         # 0x5140e48 <writev+0x5140e48>
 4c7eef9:      	cmpl	$0x1, 0x8(%rsp)
 4c7eefe:      	movl	%ebp, 0x2c(%rsp)
 4c7ef02:      	movq	%r14, 0x30(%rsp)
 4c7ef07:      	jne	0x4c7ef25 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x185>
 4c7ef09:      	movq	0x10(%rsp), %rsi
 4c7ef0e:      	testq	%rsi, %rsi
 4c7ef11:      	jle	0x4c7ef70 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x1d0>
 4c7ef13:      	movq	0x18(%rsp), %rdi
 4c7ef18:      	movl	$0x1, %edx
 4c7ef1d:      	callq	*0x4c1e1d(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7ef23:      	jmp	0x4c7ef70 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x1d0>
 4c7ef25:      	movq	0x10(%rsp), %r15
 4c7ef2a:      	cmpq	$-0x1, %r15
 4c7ef2e:      	je	0x4c7ef70 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x1d0>
 4c7ef30:      	movq	0x18(%rsp), %r12
 4c7ef35:      	movq	0x20(%rsp), %rsi
 4c7ef3a:      	movq	%r12, %rdi
 4c7ef3d:      	callq	*0x4c23d5(%rip)         # 0x5141318 <writev+0x5141318>
 4c7ef43:      	movq	%rax, %r13
 4c7ef46:      	testq	%r15, %r15
 4c7ef49:      	je	0x4c7ef5c <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x1bc>
 4c7ef4b:      	movl	$0x1, %edx
 4c7ef50:      	movq	%r12, %rdi
 4c7ef53:      	movq	%r15, %rsi
 4c7ef56:      	callq	*0x4c1de4(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7ef5c:      	testb	$0x1, %r13b
 4c7ef60:      	movl	$0x3f666666, %eax       # imm = 0x3F666666
 4c7ef65:      	jne	0x4c7ef75 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x1d5>
 4c7ef67:      	shrq	$0x20, %r13
 4c7ef6b:      	movl	%r13d, %eax
 4c7ef6e:      	jmp	0x4c7ef75 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x1d5>
 4c7ef70:      	movl	$0x3f666666, %eax       # imm = 0x3F666666
 4c7ef75:      	movl	%eax, 0x28(%rsp)
 4c7ef79:      	leaq	-0x25146b0(%rip), %rdi  # 0x276a8d0 <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x22c>
 4c7ef80:      	movl	$0x12, %esi
 4c7ef85:      	movl	$0x400, %edx            # imm = 0x400
 4c7ef8a:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7ef8f:      	movq	%rax, %r15
 4c7ef92:      	leaq	-0x25146b7(%rip), %rdi  # 0x276a8e2 <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x23e>
 4c7ef99:      	movl	$0x13, %esi
 4c7ef9e:      	movl	$0x100, %edx            # imm = 0x100
 4c7efa3:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7efa8:      	movq	%rax, %r12
 4c7efab:      	leaq	-0x25146bd(%rip), %rdi  # 0x276a8f5 <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x251>
 4c7efb2:      	movl	$0x15, %esi
 4c7efb7:      	movl	$0x100, %edx            # imm = 0x100
 4c7efbc:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7efc1:      	movq	%rax, %r13
 4c7efc4:      	leaq	-0x25146c1(%rip), %rdi  # 0x276a90a <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x266>
 4c7efcb:      	movl	$0x1a, %esi
 4c7efd0:      	movl	$0x20, %edx
 4c7efd5:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7efda:      	movq	%rax, %rbp
 4c7efdd:      	leaq	-0x25146c0(%rip), %rdi  # 0x276a924 <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x280>
 4c7efe4:      	movl	$0x14, %esi
 4c7efe9:      	movl	$0x4, %edx
 4c7efee:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7eff3:      	movq	%rax, %r14
 4c7eff6:      	leaq	-0x25146c5(%rip), %rdi  # 0x276a938 <anon.0dedcd1518bcfc2768942a657fc7c7b9.27.llvm.1391286506594828566+0x294>
 4c7effd:      	movl	$0x16, %esi
 4c7f002:      	movl	$0x8, %edx
 4c7f007:      	callq	0x4c7eae0 <_RNCNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB7_10LanePolicy8from_envs_0B7_>
 4c7f00c:      	movl	0x2c(%rsp), %ecx
 4c7f010:      	movl	%ecx, 0x38(%rbx)
 4c7f013:      	movq	0x30(%rsp), %rcx
 4c7f018:      	movq	%rcx, (%rbx)
 4c7f01b:      	movl	$0x7f800000, 0x3c(%rbx) # imm = 0x7F800000
 4c7f022:      	movl	0x4(%rsp), %ecx
 4c7f026:      	movl	%ecx, 0x40(%rbx)
 4c7f029:      	movl	0x28(%rsp), %ecx
 4c7f02d:      	movl	%ecx, 0x44(%rbx)
 4c7f030:      	movq	%r15, 0x8(%rbx)
 4c7f034:      	movq	%r12, 0x10(%rbx)
 4c7f038:      	movq	%r13, 0x18(%rbx)
 4c7f03c:      	movq	%rbp, 0x20(%rbx)
 4c7f040:      	movq	%r14, 0x28(%rbx)
 4c7f044:      	movq	%rax, 0x30(%rbx)
 4c7f048:      	movq	%rbx, %rax
 4c7f04b:      	addq	$0x38, %rsp
 4c7f04f:      	popq	%rbx
 4c7f050:      	popq	%r12
 4c7f052:      	popq	%r13
 4c7f054:      	popq	%r14
 4c7f056:      	popq	%r15
 4c7f058:      	popq	%rbp
 4c7f059:      	retq
 4c7f05a:      	jmp	0x4c7f05c <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x2bc>
 4c7f05c:      	movq	%rax, %rbx
 4c7f05f:      	testq	%r15, %r15
 4c7f062:      	je	0x4c7f08a <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x2ea>
 4c7f064:      	movl	$0x1, %edx
 4c7f069:      	movq	%r12, %rdi
 4c7f06c:      	movq	%r15, %rsi
 4c7f06f:      	jmp	0x4c7f084 <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x2e4>
 4c7f071:      	movq	%rax, %rbx
 4c7f074:      	testq	%r14, %r14
 4c7f077:      	je	0x4c7f08a <_RNvMs0_CsdMwdNqnNPrU_11memra_lanesNtB5_10LanePolicy8from_env+0x2ea>
 4c7f079:      	movl	$0x1, %edx
 4c7f07e:      	movq	%r15, %rdi
 4c7f081:      	movq	%r14, %rsi
 4c7f084:      	callq	*0x4c1cb6(%rip)         # 0x5140d40 <writev+0x5140d40>
 4c7f08a:      	movq	%rbx, %rdi
 4c7f08d:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 4c7f092:      	int3
 4c7f093:      	int3
 4c7f094:      	int3
 4c7f095:      	int3
 4c7f096:      	int3
 4c7f097:      	int3
 4c7f098:      	int3
 4c7f099:      	int3
 4c7f09a:      	int3
 4c7f09b:      	int3
 4c7f09c:      	int3
 4c7f09d:      	int3
 4c7f09e:      	int3
 4c7f09f:      	int3

0000000004df5344 <_RINvNtCs4NRVxsYgnAr_4core9panicking13assert_failedhhECsMu17NaGdWQ_12aho_corasick>:
 4df5344:      	subq	$0x18, %rsp
 4df5348:      	movq	%r9, %rax
 4df534b:      	movq	%r8, %r10
 4df534e:      	movq	%rcx, %r9
 4df5351:      	leaq	0x8(%rsp), %r8
 4df5356:      	movq	%rsi, (%r8)
 4df5359:      	leaq	0x10(%rsp), %rcx
 4df535e:      	movq	%rdx, (%rcx)
 4df5361:      	leaq	0x30f040(%rip), %rdx    # 0x51043a8 <anon.bf29f5896d2c2b3f8e4093eb8a1dd845.6.llvm.10101584134372580637+0xc8>
 4df5368:      	movq	%r8, %rsi
 4df536b:      	movq	%rdx, %r8
 4df536e:      	pushq	%rax
 4df536f:      	pushq	%r10
 4df5371:      	callq	*0x34ee51(%rip)         # 0x51441c8 <writev+0x51441c8>
 4df5377:      	int3
 4df5378:      	int3
 4df5379:      	int3
 4df537a:      	int3
 4df537b:      	int3
 4df537c:      	int3
 4df537d:      	int3
 4df537e:      	int3
 4df537f:      	int3

0000000004f81520 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCscdodAO9FK5_5alloc6string6StringECs2AWtUsOyxgP_3std>:
 4f81520:      	movq	(%rdi), %rsi
 4f81523:      	testq	%rsi, %rsi
 4f81526:      	je	0x4f81537 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCscdodAO9FK5_5alloc6string6StringECs2AWtUsOyxgP_3std+0x17>
 4f81528:      	movq	0x8(%rdi), %rdi
 4f8152c:      	movl	$0x1, %edx
 4f81531:      	jmpq	*0x1bf809(%rip)         # 0x5140d40 <writev+0x5140d40>
 4f81537:      	retq
 4f81538:      	int3
 4f81539:      	int3
 4f8153a:      	int3
 4f8153b:      	int3
 4f8153c:      	int3
 4f8153d:      	int3
 4f8153e:      	int3
 4f8153f:      	int3

0000000004f82020 <_RINvNtCs4NRVxsYgnAr_4core9panicking13assert_failedllECs2AWtUsOyxgP_3std>:
 4f82020:      	pushq	%rbp
 4f82021:      	movq	%rsp, %rbp
 4f82024:      	subq	$0x10, %rsp
 4f82028:      	movq	%r9, %rax
 4f8202b:      	movq	%r8, %r10
 4f8202e:      	movq	%rcx, %r9
 4f82031:      	leaq	-0x8(%rbp), %r8
 4f82035:      	movq	%rsi, (%r8)
 4f82038:      	leaq	-0x10(%rbp), %rcx
 4f8203c:      	movq	%rdx, (%rcx)
 4f8203f:      	leaq	0x1bba92(%rip), %rdx    # 0x513dad8 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x4a0>
 4f82046:      	movq	%r8, %rsi
 4f82049:      	movq	%rdx, %r8
 4f8204c:      	pushq	%rax
 4f8204d:      	pushq	%r10
 4f8204f:      	callq	*0x1c2173(%rip)         # 0x51441c8 <writev+0x51441c8>

0000000004f82a80 <_RINvNtNtCs2AWtUsOyxgP_3std3sys9backtrace26___rust_end_short_backtraceNCNvNtB6_5alloc8rust_oom0zEB6_>:
 4f82a80:      	pushq	%rbp
 4f82a81:      	movq	%rsp, %rbp
 4f82a84:      	callq	0x4f8e170 <_RNCNvNtCs2AWtUsOyxgP_3std5alloc8rust_oom0B5_>
 4f82a89:      	int3
 4f82a8a:      	int3
 4f82a8b:      	int3
 4f82a8c:      	int3
 4f82a8d:      	int3
 4f82a8e:      	int3
 4f82a8f:      	int3

0000000004f82a90 <_RINvNtNtCs2AWtUsOyxgP_3std3sys9backtrace26___rust_end_short_backtraceNCNvNtB6_9panicking13panic_handler0zEB6_>:
 4f82a90:      	pushq	%rbp
 4f82a91:      	movq	%rsp, %rbp
 4f82a94:      	callq	0x4f8e300 <_RNCNvNtCs2AWtUsOyxgP_3std9panicking13panic_handler0B5_>
 4f82a99:      	int3
 4f82a9a:      	int3
 4f82a9b:      	int3
 4f82a9c:      	int3
 4f82a9d:      	int3
 4f82a9e:      	int3
 4f82a9f:      	int3

0000000004f8f600 <_RNvCs9wFQrvczXsK_7___rustc17rust_begin_unwind>:
 4f8f600:      	pushq	%rbp
 4f8f601:      	movq	%rsp, %rbp
 4f8f604:      	subq	$0x20, %rsp
 4f8f608:      	movups	(%rdi), %xmm0
 4f8f60b:      	movups	%xmm0, -0x18(%rbp)
 4f8f60f:      	movq	%rdi, -0x8(%rbp)
 4f8f613:      	leaq	-0x18(%rbp), %rdi
 4f8f617:      	callq	*0x1bc7cb(%rip)         # 0x514bde8 <writev+0x514bde8>
 4f8f61d:      	int3
 4f8f61e:      	int3
 4f8f61f:      	int3

0000000004f8f6b0 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception>:
 4f8f6b0:      	pushq	%rbp
 4f8f6b1:      	movq	%rsp, %rbp
 4f8f6b4:      	pushq	%r15
 4f8f6b6:      	pushq	%r14
 4f8f6b8:      	pushq	%r13
 4f8f6ba:      	pushq	%r12
 4f8f6bc:      	pushq	%rbx
 4f8f6bd:      	subq	$0x18, %rsp
 4f8f6c1:      	movl	$0x44, %ebx
 4f8f6c6:      	leaq	-0x27af833(%rip), %r14  # 0x27dfe9a <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x203b>
 4f8f6cd:      	leaq	-0x30(%rbp), %r15
 4f8f6d1:      	movq	0x1bad70(%rip), %r13    # 0x514a448 <writev+0x514a448>
 4f8f6d8:      	movq	0x1b2fb9(%rip), %r12    # 0x5142698 <writev+0x5142698>
 4f8f6df:      	jmp	0x4f8f6f9 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0x49>
 4f8f6e1:      	nopw	%cs:(%rax,%rax)
 4f8f6f0:      	testq	%rbx, %rbx
 4f8f6f3:      	je	0x4f8f786 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0xd6>
 4f8f6f9:      	movl	$0x2, %edi
 4f8f6fe:      	movq	%r14, %rsi
 4f8f701:      	movq	%rbx, %rdx
 4f8f704:      	callq	*%r13
 4f8f707:      	cmpq	$-0x1, %rax
 4f8f70b:      	je	0x4f8f730 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0x80>
 4f8f70d:      	testq	%rax, %rax
 4f8f710:      	je	0x4f8f772 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0xc2>
 4f8f712:      	movq	%rbx, %rcx
 4f8f715:      	subq	%rax, %rcx
 4f8f718:      	jb	0x4f8f75a <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0xaa>
 4f8f71a:      	addq	%rax, %r14
 4f8f71d:      	movq	%rcx, %rbx
 4f8f720:      	jmp	0x4f8f6f0 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0x40>
 4f8f722:      	nopw	%cs:(%rax,%rax)
 4f8f730:      	callq	*%r12
 4f8f733:      	movl	(%rax), %eax
 4f8f735:      	movl	%eax, %ecx
 4f8f737:      	shlq	$0x20, %rax
 4f8f73b:      	orq	$0x2, %rax
 4f8f73f:      	movq	$0x1, -0x38(%rbp)
 4f8f747:      	movq	%rax, -0x30(%rbp)
 4f8f74b:      	cmpl	$0x4, %ecx
 4f8f74e:      	jne	0x4f8f779 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0xc9>
 4f8f750:      	movq	%r15, %rdi
 4f8f753:      	callq	0x4f81540 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorEBH_>
 4f8f758:      	jmp	0x4f8f6f0 <_RNvCs9wFQrvczXsK_7___rustc24___rust_foreign_exception+0x40>
 4f8f75a:      	leaq	0x1af9ff(%rip), %rcx    # 0x513f160 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1b28>
 4f8f761:      	movq	%rax, %rdi
 4f8f764:      	movq	%rbx, %rsi
 4f8f767:      	movq	%rbx, %rdx
 4f8f76a:      	callq	*0x1b1700(%rip)         # 0x5140e70 <writev+0x5140e70>
 4f8f770:      	ud2
 4f8f772:      	leaq	0x1aef4f(%rip), %rax    # 0x513e6c8 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1090>
 4f8f779:      	movq	%rax, -0x38(%rbp)
 4f8f77d:      	leaq	-0x38(%rbp), %rdi
 4f8f781:      	callq	0x4f81540 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorEBH_>
 4f8f786:      	callq	*0x1b248c(%rip)         # 0x5141c18 <writev+0x5141c18>
 4f8f78c:      	callq	*0x1b168e(%rip)         # 0x5140e20 <writev+0x5140e20>
 4f8f792:      	callq	*0x1b1688(%rip)         # 0x5140e20 <writev+0x5140e20>
 4f8f798:      	int3
 4f8f799:      	int3
 4f8f79a:      	int3
 4f8f79b:      	int3
 4f8f79c:      	int3
 4f8f79d:      	int3
 4f8f79e:      	int3
 4f8f79f:      	int3

0000000004f8f7a0 <_RNvCs9wFQrvczXsK_7___rustc26___rust_alloc_error_handler>:
 4f8f7a0:      	pushq	%rbp
 4f8f7a1:      	movq	%rsp, %rbp
 4f8f7a4:      	movq	%rdi, %rax
 4f8f7a7:      	movq	%rsi, %rdi
 4f8f7aa:      	movq	%rax, %rsi
 4f8f7ad:      	callq	*0x1bc775(%rip)         # 0x514bf28 <writev+0x514bf28>
 4f8f7b3:      	int3
 4f8f7b4:      	int3
 4f8f7b5:      	int3
 4f8f7b6:      	int3
 4f8f7b7:      	int3
 4f8f7b8:      	int3
 4f8f7b9:      	int3
 4f8f7ba:      	int3
 4f8f7bb:      	int3
 4f8f7bc:      	int3
 4f8f7bd:      	int3
 4f8f7be:      	int3
 4f8f7bf:      	int3

0000000004f8f840 <_RNvMNtCs2AWtUsOyxgP_3std4timeNtB2_7Instant25saturating_duration_since>:
 4f8f840:      	pushq	%rbp
 4f8f841:      	movq	%rsp, %rbp
 4f8f844:      	subq	$0x30, %rsp
 4f8f848:      	movq	%rdi, %rax
 4f8f84b:      	movq	%rsi, -0x10(%rbp)
 4f8f84f:      	movl	%edx, -0x8(%rbp)
 4f8f852:      	leaq	-0x28(%rbp), %rdi
 4f8f856:      	leaq	-0x10(%rbp), %rdx
 4f8f85a:      	movq	%rax, %rsi
 4f8f85d:      	callq	0x4f91600 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys3pal4unix4timeNtB2_8Timespec12sub_timespec>
 4f8f862:      	xorl	%eax, %eax
 4f8f864:      	cmpb	$0x0, -0x28(%rbp)
 4f8f868:      	movl	-0x18(%rbp), %edx
 4f8f86b:      	cmovnel	%eax, %edx
 4f8f86e:      	cmoveq	-0x20(%rbp), %rax
 4f8f873:      	addq	$0x30, %rsp
 4f8f877:      	popq	%rbp
 4f8f878:      	retq
 4f8f879:      	int3
 4f8f87a:      	int3
 4f8f87b:      	int3
 4f8f87c:      	int3
 4f8f87d:      	int3
 4f8f87e:      	int3
 4f8f87f:      	int3

0000000004f8f880 <_RNvMNtCs2AWtUsOyxgP_3std4timeNtB2_7Instant3now>:
 4f8f880:      	movl	$0x1, %edi
 4f8f885:      	jmp	0x4f916d0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys3pal4unix4timeNtB2_8Timespec3now>
 4f8f88a:      	int3
 4f8f88b:      	int3
 4f8f88c:      	int3
 4f8f88d:      	int3
 4f8f88e:      	int3
 4f8f88f:      	int3

0000000004f8f890 <_RNvMNtCs2AWtUsOyxgP_3std4timeNtB2_7Instant7elapsed>:
 4f8f890:      	pushq	%rbp
 4f8f891:      	movq	%rsp, %rbp
 4f8f894:      	pushq	%rbx
 4f8f895:      	subq	$0x38, %rsp
 4f8f899:      	movq	%rdi, %rbx
 4f8f89c:      	movl	$0x1, %edi
 4f8f8a1:      	callq	0x4f916d0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys3pal4unix4timeNtB2_8Timespec3now>
 4f8f8a6:      	movq	(%rbx), %rcx
 4f8f8a9:      	movl	0x8(%rbx), %esi
 4f8f8ac:      	movq	%rax, -0x18(%rbp)
 4f8f8b0:      	movl	%edx, -0x10(%rbp)
 4f8f8b3:      	movq	%rcx, -0x28(%rbp)
 4f8f8b7:      	movl	%esi, -0x20(%rbp)
 4f8f8ba:      	leaq	-0x40(%rbp), %rdi
 4f8f8be:      	leaq	-0x18(%rbp), %rsi
 4f8f8c2:      	leaq	-0x28(%rbp), %rdx
 4f8f8c6:      	callq	0x4f91600 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys3pal4unix4timeNtB2_8Timespec12sub_timespec>
 4f8f8cb:      	xorl	%eax, %eax
 4f8f8cd:      	cmpb	$0x0, -0x40(%rbp)
 4f8f8d1:      	movl	-0x30(%rbp), %edx
 4f8f8d4:      	cmovnel	%eax, %edx
 4f8f8d7:      	cmoveq	-0x38(%rbp), %rax
 4f8f8dc:      	addq	$0x38, %rsp
 4f8f8e0:      	popq	%rbx
 4f8f8e1:      	popq	%rbp
 4f8f8e2:      	retq
 4f8f8e3:      	int3
 4f8f8e4:      	int3
 4f8f8e5:      	int3
 4f8f8e6:      	int3
 4f8f8e7:      	int3
 4f8f8e8:      	int3
 4f8f8e9:      	int3
 4f8f8ea:      	int3
 4f8f8eb:      	int3
 4f8f8ec:      	int3
 4f8f8ed:      	int3
 4f8f8ee:      	int3
 4f8f8ef:      	int3

0000000004f917a0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended>:
 4f917a0:      	pushq	%rbp
 4f917a1:      	movq	%rsp, %rbp
 4f917a4:      	pushq	%r15
 4f917a6:      	pushq	%r14
 4f917a8:      	pushq	%r12
 4f917aa:      	pushq	%rbx
 4f917ab:      	subq	$0x20, %rsp
 4f917af:      	movq	%rdi, %rbx
 4f917b2:      	movl	(%rdi), %eax
 4f917b4:      	cmpl	$0x1, %eax
 4f917b7:      	jne	0x4f917d0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x30>
 4f917b9:      	movl	$0xffffff9d, %ecx       # imm = 0xFFFFFF9D
 4f917be:      	nop
 4f917c0:      	pause
 4f917c2:      	movl	(%rbx), %eax
 4f917c4:      	cmpl	$0x1, %eax
 4f917c7:      	jne	0x4f917d0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x30>
 4f917c9:      	testl	%ecx, %ecx
 4f917cb:      	leal	0x1(%rcx), %ecx
 4f917ce:      	jne	0x4f917c0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x20>
 4f917d0:      	testl	%eax, %eax
 4f917d2:      	jne	0x4f917ee <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x4e>
 4f917d4:      	movl	$0x1, %ecx
 4f917d9:      	xorl	%eax, %eax
 4f917db:      	lock
 4f917dc:      	cmpxchgl	%ecx, (%rbx)
 4f917df:      	jne	0x4f917ee <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x4e>
 4f917e1:      	addq	$0x20, %rsp
 4f917e5:      	popq	%rbx
 4f917e6:      	popq	%r12
 4f917e8:      	popq	%r14
 4f917ea:      	popq	%r15
 4f917ec:      	popq	%rbp
 4f917ed:      	retq
 4f917ee:      	leaq	-0x30(%rbp), %r14
 4f917f2:      	movq	0x1b21af(%rip), %r15    # 0x51439a8 <writev+0x51439a8>
 4f917f9:      	movq	0x1b0e98(%rip), %r12    # 0x5142698 <writev+0x5142698>
 4f91800:      	cmpl	$0x2, %eax
 4f91803:      	je	0x4f91810 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x70>
 4f91805:      	movl	$0x2, %eax
 4f9180a:      	xchgl	%eax, (%rbx)
 4f9180c:      	testl	%eax, %eax
 4f9180e:      	je	0x4f917e1 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x41>
 4f91810:      	movq	$0x0, -0x38(%rbp)
 4f91818:      	nopl	(%rax,%rax)
 4f91820:      	movl	(%rbx), %eax
 4f91822:      	cmpl	$0x2, %eax
 4f91825:      	jne	0x4f91863 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0xc3>
 4f91827:      	cmpb	$0x0, -0x38(%rbp)
 4f9182b:      	movl	$0x0, %r8d
 4f91831:      	cmovneq	%r14, %r8
 4f91835:      	movl	$0xffffffff, (%rsp)     # imm = 0xFFFFFFFF
 4f9183c:      	movl	$0xca, %edi
 4f91841:      	movq	%rbx, %rsi
 4f91844:      	movl	$0x89, %edx
 4f91849:      	movl	$0x2, %ecx
 4f9184e:      	xorl	%r9d, %r9d
 4f91851:      	xorl	%eax, %eax
 4f91853:      	callq	*%r15
 4f91856:      	testq	%rax, %rax
 4f91859:      	jns	0x4f91863 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0xc3>
 4f9185b:      	callq	*%r12
 4f9185e:      	cmpl	$0x4, (%rax)
 4f91861:      	je	0x4f91820 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x80>
 4f91863:      	movl	(%rbx), %eax
 4f91865:      	cmpl	$0x1, %eax
 4f91868:      	jne	0x4f91800 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x60>
 4f9186a:      	movl	$0xffffff9d, %ecx       # imm = 0xFFFFFF9D
 4f9186f:      	nop
 4f91870:      	pause
 4f91872:      	movl	(%rbx), %eax
 4f91874:      	leal	0x1(%rcx), %edx
 4f91877:      	cmpl	$0x1, %eax
 4f9187a:      	jne	0x4f91800 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x60>
 4f9187c:      	testl	%ecx, %ecx
 4f9187e:      	movl	%edx, %ecx
 4f91880:      	jne	0x4f91870 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0xd0>
 4f91882:      	jmp	0x4f91800 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex14lock_contended+0x60>
 4f91887:      	int3
 4f91888:      	int3
 4f91889:      	int3
 4f9188a:      	int3
 4f9188b:      	int3
 4f9188c:      	int3
 4f9188d:      	int3
 4f9188e:      	int3
 4f9188f:      	int3

0000000004f91890 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync5mutex5futexNtB2_5Mutex4wake>:
 4f91890:      	movq	%rdi, %rsi
 4f91893:      	movl	$0xca, %edi
 4f91898:      	movl	$0x81, %edx
 4f9189d:      	movl	$0x1, %ecx
 4f918a2:      	xorl	%eax, %eax
 4f918a4:      	jmpq	*0x1b20fe(%rip)         # 0x51439a8 <writev+0x51439a8>
 4f918aa:      	int3
 4f918ab:      	int3
 4f918ac:      	int3
 4f918ad:      	int3
 4f918ae:      	int3
 4f918af:      	int3

0000000004f91a50 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended>:
 4f91a50:      	pushq	%rbp
 4f91a51:      	movq	%rsp, %rbp
 4f91a54:      	pushq	%r15
 4f91a56:      	pushq	%r14
 4f91a58:      	pushq	%r13
 4f91a5a:      	pushq	%r12
 4f91a5c:      	pushq	%rbx
 4f91a5d:      	subq	$0x28, %rsp
 4f91a61:      	movq	%rdi, %rbx
 4f91a64:      	movl	(%rdi), %eax
 4f91a66:      	testl	$0x3fffffff, %eax       # imm = 0x3FFFFFFF
 4f91a6b:      	sete	%cl
 4f91a6e:      	testl	%eax, %eax
 4f91a70:      	sets	%dl
 4f91a73:      	orb	%cl, %dl
 4f91a75:      	jne	0x4f91a99 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x49>
 4f91a77:      	movl	$0xffffff9d, %ecx       # imm = 0xFFFFFF9D
 4f91a7c:      	nopl	(%rax)
 4f91a80:      	pause
 4f91a82:      	movl	(%rbx), %eax
 4f91a84:      	testl	%eax, %eax
 4f91a86:      	js	0x4f91a99 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x49>
 4f91a88:      	movl	%eax, %edx
 4f91a8a:      	andl	$0x3fffffff, %edx       # imm = 0x3FFFFFFF
 4f91a90:      	je	0x4f91a99 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x49>
 4f91a92:      	testl	%ecx, %ecx
 4f91a94:      	leal	0x1(%rcx), %ecx
 4f91a97:      	jne	0x4f91a80 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x30>
 4f91a99:      	leaq	0x4(%rbx), %r14
 4f91a9d:      	movl	$0x3fffffff, %r12d      # imm = 0x3FFFFFFF
 4f91aa3:      	movq	0x1b1efe(%rip), %r13    # 0x51439a8 <writev+0x51439a8>
 4f91aaa:      	jmp	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91aac:      	nopl	(%rax)
 4f91ab0:      	movl	%eax, %ecx
 4f91ab2:      	orl	%r12d, %ecx
 4f91ab5:      	lock
 4f91ab6:      	cmpxchgl	%ecx, (%rbx)
 4f91ab9:      	je	0x4f91b9a <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x14a>
 4f91abf:      	testl	$0x3fffffff, %eax       # imm = 0x3FFFFFFF
 4f91ac4:      	je	0x4f91ab0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x60>
 4f91ac6:      	testl	%eax, %eax
 4f91ac8:      	js	0x4f91ad8 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x88>
 4f91aca:      	movl	%eax, %ecx
 4f91acc:      	orl	$0x80000000, %ecx       # imm = 0x80000000
 4f91ad2:      	lock
 4f91ad3:      	cmpxchgl	%ecx, (%rbx)
 4f91ad6:      	jne	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91ad8:      	movl	0x4(%rbx), %r15d
 4f91adc:      	movl	(%rbx), %eax
 4f91ade:      	testl	$0x3fffffff, %eax       # imm = 0x3FFFFFFF
 4f91ae3:      	sete	%cl
 4f91ae6:      	testl	%eax, %eax
 4f91ae8:      	setns	%dl
 4f91aeb:      	orb	%cl, %dl
 4f91aed:      	movl	$0xbfffffff, %r12d      # imm = 0xBFFFFFFF
 4f91af3:      	jne	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91af5:      	movq	$0x0, -0x40(%rbp)
 4f91afd:      	nopl	(%rax)
 4f91b00:      	movl	(%r14), %eax
 4f91b03:      	cmpl	%r15d, %eax
 4f91b06:      	jne	0x4f91b49 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0xf9>
 4f91b08:      	cmpb	$0x0, -0x40(%rbp)
 4f91b0c:      	movl	$0x0, %r8d
 4f91b12:      	leaq	-0x38(%rbp), %rax
 4f91b16:      	cmovneq	%rax, %r8
 4f91b1a:      	movl	$0xffffffff, (%rsp)     # imm = 0xFFFFFFFF
 4f91b21:      	movl	$0xca, %edi
 4f91b26:      	movq	%r14, %rsi
 4f91b29:      	movl	$0x89, %edx
 4f91b2e:      	movl	%r15d, %ecx
 4f91b31:      	xorl	%r9d, %r9d
 4f91b34:      	xorl	%eax, %eax
 4f91b36:      	callq	*%r13
 4f91b39:      	testq	%rax, %rax
 4f91b3c:      	jns	0x4f91b49 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0xf9>
 4f91b3e:      	callq	*0x1b0b54(%rip)         # 0x5142698 <writev+0x5142698>
 4f91b44:      	cmpl	$0x4, (%rax)
 4f91b47:      	je	0x4f91b00 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0xb0>
 4f91b49:      	movl	(%rbx), %eax
 4f91b4b:      	testl	$0x3fffffff, %eax       # imm = 0x3FFFFFFF
 4f91b50:      	sete	%cl
 4f91b53:      	testl	%eax, %eax
 4f91b55:      	sets	%dl
 4f91b58:      	orb	%cl, %dl
 4f91b5a:      	jne	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91b60:      	movl	$0xffffff9d, %ecx       # imm = 0xFFFFFF9D
 4f91b65:      	nopw	%cs:(%rax,%rax)
 4f91b70:      	pause
 4f91b72:      	movl	(%rbx), %eax
 4f91b74:      	movl	%eax, %esi
 4f91b76:      	andl	$0x3fffffff, %esi       # imm = 0x3FFFFFFF
 4f91b7c:      	leal	0x1(%rcx), %edx
 4f91b7f:      	testl	%eax, %eax
 4f91b81:      	js	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91b87:      	testl	%esi, %esi
 4f91b89:      	je	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91b8f:      	testl	%ecx, %ecx
 4f91b91:      	movl	%edx, %ecx
 4f91b93:      	jne	0x4f91b70 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x120>
 4f91b95:      	jmp	0x4f91abf <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock15write_contended+0x6f>
 4f91b9a:      	addq	$0x28, %rsp
 4f91b9e:      	popq	%rbx
 4f91b9f:      	popq	%r12
 4f91ba1:      	popq	%r13
 4f91ba3:      	popq	%r14
 4f91ba5:      	popq	%r15
 4f91ba7:      	popq	%rbp
 4f91ba8:      	retq
 4f91ba9:      	int3
 4f91baa:      	int3
 4f91bab:      	int3
 4f91bac:      	int3
 4f91bad:      	int3
 4f91bae:      	int3
 4f91baf:      	int3

0000000004f91bb0 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers>:
 4f91bb0:      	pushq	%rbp
 4f91bb1:      	movq	%rsp, %rbp
 4f91bb4:      	pushq	%rbx
 4f91bb5:      	pushq	%rax
 4f91bb6:      	testl	$0x3fffffff, %esi       # imm = 0x3FFFFFFF
 4f91bbc:      	jne	0x4f91c6e <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0xbe>
 4f91bc2:      	movq	%rdi, %rbx
 4f91bc5:      	movl	%esi, %eax
 4f91bc7:      	negl	%eax
 4f91bc9:      	jno	0x4f91bf6 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0x46>
 4f91bcb:      	xorl	%ecx, %ecx
 4f91bcd:      	movl	$0x80000000, %eax       # imm = 0x80000000
 4f91bd2:      	lock
 4f91bd3:      	cmpxchgl	%ecx, (%rbx)
 4f91bd6:      	jne	0x4f91bf4 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0x44>
 4f91bd8:      	lock
 4f91bd9:      	incl	0x4(%rbx)
 4f91bdc:      	addq	$0x4, %rbx
 4f91be0:      	movl	$0xca, %edi
 4f91be5:      	movq	%rbx, %rsi
 4f91be8:      	movl	$0x81, %edx
 4f91bed:      	movl	$0x1, %ecx
 4f91bf2:      	jmp	0x4f91c59 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0xa9>
 4f91bf4:      	movl	%eax, %esi
 4f91bf6:      	cmpl	$0x40000000, %esi       # imm = 0x40000000
 4f91bfc:      	je	0x4f91c3a <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0x8a>
 4f91bfe:      	cmpl	$0xc0000000, %esi       # imm = 0xC0000000
 4f91c04:      	jne	0x4f91c67 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0xb7>
 4f91c06:      	movl	$0x40000000, %ecx       # imm = 0x40000000
 4f91c0b:      	movl	$0xc0000000, %eax       # imm = 0xC0000000
 4f91c10:      	lock
 4f91c11:      	cmpxchgl	%ecx, (%rbx)
 4f91c14:      	jne	0x4f91c67 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0xb7>
 4f91c16:      	leaq	0x4(%rbx), %rsi
 4f91c1a:      	lock
 4f91c1b:      	incl	0x4(%rbx)
 4f91c1e:      	movl	$0xca, %edi
 4f91c23:      	movl	$0x81, %edx
 4f91c28:      	movl	$0x1, %ecx
 4f91c2d:      	xorl	%eax, %eax
 4f91c2f:      	callq	*0x1b1d73(%rip)         # 0x51439a8 <writev+0x51439a8>
 4f91c35:      	testq	%rax, %rax
 4f91c38:      	jg	0x4f91c67 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0xb7>
 4f91c3a:      	xorl	%ecx, %ecx
 4f91c3c:      	movl	$0x40000000, %eax       # imm = 0x40000000
 4f91c41:      	lock
 4f91c42:      	cmpxchgl	%ecx, (%rbx)
 4f91c45:      	jne	0x4f91c67 <_RNvMNtNtNtNtCs2AWtUsOyxgP_3std3sys4sync6rwlock5futexNtB2_6RwLock22wake_writer_or_readers+0xb7>
 4f91c47:      	movl	$0xca, %edi
 4f91c4c:      	movq	%rbx, %rsi
 4f91c4f:      	movl	$0x81, %edx
 4f91c54:      	movl	$0x7fffffff, %ecx       # imm = 0x7FFFFFFF
 4f91c59:      	xorl	%eax, %eax
 4f91c5b:      	addq	$0x8, %rsp
 4f91c5f:      	popq	%rbx
 4f91c60:      	popq	%rbp
 4f91c61:      	jmpq	*0x1b1d41(%rip)         # 0x51439a8 <writev+0x51439a8>
 4f91c67:      	addq	$0x8, %rsp
 4f91c6b:      	popq	%rbx
 4f91c6c:      	popq	%rbp
 4f91c6d:      	retq
 4f91c6e:      	leaq	-0x27b1bfa(%rip), %rdi  # 0x27e007b <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x221c>
 4f91c75:      	leaq	0x1ac394(%rip), %rdx    # 0x513e010 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x9d8>
 4f91c7c:      	movl	$0x24, %esi
 4f91c81:      	callq	*0x1af281(%rip)         # 0x5140f08 <writev+0x5140f08>
 4f91c87:      	int3
 4f91c88:      	int3
 4f91c89:      	int3
 4f91c8a:      	int3
 4f91c8b:      	int3
 4f91c8c:      	int3
 4f91c8d:      	int3
 4f91c8e:      	int3
 4f91c8f:      	int3

0000000004f940a0 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call>:
 4f940a0:      	pushq	%rbp
 4f940a1:      	movq	%rsp, %rbp
 4f940a4:      	pushq	%r15
 4f940a6:      	pushq	%r14
 4f940a8:      	pushq	%r13
 4f940aa:      	pushq	%r12
 4f940ac:      	pushq	%rbx
 4f940ad:      	subq	$0x38, %rsp
 4f940b1:      	movq	%rdi, %rbx
 4f940b4:      	movl	(%rdi), %r12d
 4f940b7:      	leaq	-0x38(%rbp), %r13
 4f940bb:      	testl	%esi, %esi
 4f940bd:      	movq	%rcx, -0x50(%rbp)
 4f940c1:      	movq	%rdx, -0x48(%rbp)
 4f940c5:      	je	0x4f941db <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x13b>
 4f940cb:      	movq	0x1af8d6(%rip), %r14    # 0x51439a8 <writev+0x51439a8>
 4f940d2:      	movq	0x1ae5bf(%rip), %r15    # 0x5142698 <writev+0x5142698>
 4f940d9:      	jmp	0x4f940e3 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x43>
 4f940db:      	nopl	(%rax,%rax)
 4f940e0:      	movl	(%rbx), %r12d
 4f940e3:      	movl	%r12d, %ecx
 4f940e6:      	andl	$0x3, %ecx
 4f940e9:      	movl	%r12d, %edx
 4f940ec:      	andl	$0x4, %edx
 4f940ef:      	leal	-0x2(%rcx), %eax
 4f940f2:      	cmpl	$0x2, %eax
 4f940f5:      	jae	0x4f94110 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x70>
 4f940f7:      	orl	$0x1, %edx
 4f940fa:      	movl	%r12d, %eax
 4f940fd:      	lock
 4f940fe:      	cmpxchgl	%edx, (%rbx)
 4f94101:      	je	0x4f9418a <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0xea>
 4f94107:      	movl	%eax, %r12d
 4f9410a:      	jmp	0x4f940e3 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x43>
 4f9410c:      	nopl	(%rax)
 4f94110:      	cmpl	$0x1, %ecx
 4f94113:      	jne	0x4f941cc <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x12c>
 4f94119:      	testl	%edx, %edx
 4f9411b:      	jne	0x4f9412d <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x8d>
 4f9411d:      	orl	$0x4, %r12d
 4f94121:      	movl	$0x1, %eax
 4f94126:      	lock
 4f94127:      	cmpxchgl	%r12d, (%rbx)
 4f9412b:      	jne	0x4f94107 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x67>
 4f9412d:      	movq	$0x0, -0x40(%rbp)
 4f94135:      	nopw	%cs:(%rax,%rax)
 4f94140:      	movl	(%rbx), %eax
 4f94142:      	cmpl	%r12d, %eax
 4f94145:      	jne	0x4f940e0 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x40>
 4f94147:      	cmpb	$0x0, -0x40(%rbp)
 4f9414b:      	movl	$0x0, %r8d
 4f94151:      	cmovneq	%r13, %r8
 4f94155:      	movl	$0xffffffff, (%rsp)     # imm = 0xFFFFFFFF
 4f9415c:      	movl	$0xca, %edi
 4f94161:      	movq	%rbx, %rsi
 4f94164:      	movl	$0x89, %edx
 4f94169:      	movl	%r12d, %ecx
 4f9416c:      	xorl	%r9d, %r9d
 4f9416f:      	xorl	%eax, %eax
 4f94171:      	callq	*%r14
 4f94174:      	testq	%rax, %rax
 4f94177:      	jns	0x4f940e0 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x40>
 4f9417d:      	callq	*%r15
 4f94180:      	cmpl	$0x4, (%rax)
 4f94183:      	je	0x4f94140 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0xa0>
 4f94185:      	jmp	0x4f940e0 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x40>
 4f9418a:      	cmpl	$0x2, %ecx
 4f9418d:      	sete	%al
 4f94190:      	movl	$0x0, -0x40(%rbp)
 4f94197:      	movb	%al, -0x3c(%rbp)
 4f9419a:      	leaq	-0x40(%rbp), %rsi
 4f9419e:      	movq	-0x48(%rbp), %rdi
 4f941a2:      	movq	-0x50(%rbp), %rax
 4f941a6:      	callq	*0x20(%rax)
 4f941a9:      	movl	-0x40(%rbp), %eax
 4f941ac:      	xchgl	%eax, (%rbx)
 4f941ae:      	testb	$0x4, %al
 4f941b0:      	je	0x4f941cc <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x12c>
 4f941b2:      	movl	$0xca, %edi
 4f941b7:      	movq	%rbx, %rsi
 4f941ba:      	movl	$0x81, %edx
 4f941bf:      	movl	$0x7fffffff, %ecx       # imm = 0x7FFFFFFF
 4f941c4:      	xorl	%eax, %eax
 4f941c6:      	callq	*0x1af7dc(%rip)         # 0x51439a8 <writev+0x51439a8>
 4f941cc:      	addq	$0x38, %rsp
 4f941d0:      	popq	%rbx
 4f941d1:      	popq	%r12
 4f941d3:      	popq	%r13
 4f941d5:      	popq	%r14
 4f941d7:      	popq	%r15
 4f941d9:      	popq	%rbp
 4f941da:      	retq
 4f941db:      	movl	%r12d, %eax
 4f941de:      	andl	$0x3, %eax
 4f941e1:      	movl	%r12d, %ecx
 4f941e4:      	andl	$0x4, %ecx
 4f941e7:      	leaq	-0x27b56de(%rip), %r15  # 0x27deb10 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0xcb1>
 4f941ee:      	movslq	(%r15,%rax,4), %rax
 4f941f2:      	addq	%r15, %rax
 4f941f5:      	jmpq	*%rax
 4f941f7:      	movq	%rax, %r14
 4f941fa:      	movl	$0x2, %eax
 4f941ff:      	xchgl	%eax, (%rbx)
 4f94201:      	testb	$0x4, %al
 4f94203:      	je	0x4f9421f <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x17f>
 4f94205:      	movl	$0xca, %edi
 4f9420a:      	movq	%rbx, %rsi
 4f9420d:      	movl	$0x81, %edx
 4f94212:      	movl	$0x7fffffff, %ecx       # imm = 0x7FFFFFFF
 4f94217:      	xorl	%eax, %eax
 4f94219:      	callq	*0x1af789(%rip)         # 0x51439a8 <writev+0x51439a8>
 4f9421f:      	movq	%r14, %rdi
 4f94222:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 4f94227:      	orl	$0x1, %ecx
 4f9422a:      	movl	%r12d, %eax
 4f9422d:      	lock
 4f9422e:      	cmpxchgl	%ecx, (%rbx)
 4f94231:      	je	0x4f942e2 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x242>
 4f94237:      	movl	%eax, %r12d
 4f9423a:      	andl	$0x3, %eax
 4f9423d:      	movl	%r12d, %ecx
 4f94240:      	andl	$0x4, %ecx
 4f94243:      	movslq	(%r15,%rax,4), %rax
 4f94247:      	addq	%r15, %rax
 4f9424a:      	jmpq	*%rax
 4f9424c:      	testl	%ecx, %ecx
 4f9424e:      	jne	0x4f94260 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x1c0>
 4f94250:      	orl	$0x4, %r12d
 4f94254:      	movl	$0x1, %eax
 4f94259:      	lock
 4f9425a:      	cmpxchgl	%r12d, (%rbx)
 4f9425e:      	jne	0x4f94237 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x197>
 4f94260:      	movq	%r8, %r14
 4f94263:      	movq	$0x0, -0x40(%rbp)
 4f9426b:      	movl	(%rbx), %eax
 4f9426d:      	cmpl	%r12d, %eax
 4f94270:      	jne	0x4f942b2 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x212>
 4f94272:      	cmpb	$0x0, -0x40(%rbp)
 4f94276:      	movl	$0x0, %r8d
 4f9427c:      	cmovneq	%r13, %r8
 4f94280:      	movl	$0xffffffff, (%rsp)     # imm = 0xFFFFFFFF
 4f94287:      	movl	$0xca, %edi
 4f9428c:      	movq	%rbx, %rsi
 4f9428f:      	movl	$0x89, %edx
 4f94294:      	movl	%r12d, %ecx
 4f94297:      	xorl	%r9d, %r9d
 4f9429a:      	xorl	%eax, %eax
 4f9429c:      	callq	*0x1af706(%rip)         # 0x51439a8 <writev+0x51439a8>
 4f942a2:      	testq	%rax, %rax
 4f942a5:      	jns	0x4f942b2 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x212>
 4f942a7:      	callq	*0x1ae3eb(%rip)         # 0x5142698 <writev+0x5142698>
 4f942ad:      	cmpl	$0x4, (%rax)
 4f942b0:      	je	0x4f9426b <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0x1cb>
 4f942b2:      	movl	(%rbx), %r12d
 4f942b5:      	movl	%r12d, %eax
 4f942b8:      	andl	$0x3, %eax
 4f942bb:      	movl	%r12d, %ecx
 4f942be:      	andl	$0x4, %ecx
 4f942c1:      	movslq	(%r15,%rax,4), %rax
 4f942c5:      	addq	%r15, %rax
 4f942c8:      	movq	%r14, %r8
 4f942cb:      	jmpq	*%rax
 4f942cd:      	leaq	-0x27b40d0(%rip), %rdi  # 0x27e0204 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x23a5>
 4f942d4:      	movl	$0x55, %esi
 4f942d9:      	movq	%r8, %rdx
 4f942dc:      	callq	*0x1acb9e(%rip)         # 0x5140e80 <writev+0x5140e80>
 4f942e2:      	xorl	%eax, %eax
 4f942e4:      	jmp	0x4f94190 <_RNvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys4sync4once5futexNtB5_4Once4call+0xf0>
 4f942e9:      	int3
 4f942ea:      	int3
 4f942eb:      	int3
 4f942ec:      	int3
 4f942ed:      	int3
 4f942ee:      	int3
 4f942ef:      	int3

0000000004faa140 <_RNvNtCs2AWtUsOyxgP_3std3env4__var>:
 4faa140:      	pushq	%rbp
 4faa141:      	movq	%rsp, %rbp
 4faa144:      	pushq	%r15
 4faa146:      	pushq	%r14
 4faa148:      	pushq	%r12
 4faa14a:      	pushq	%rbx
 4faa14b:      	subq	$0x1b0, %rsp            # imm = 0x1B0
 4faa152:      	movq	%rdi, %rbx
 4faa155:      	cmpq	$0x17f, %rdx            # imm = 0x17F
 4faa15c:      	ja	0x4faa1d4 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0x94>
 4faa15e:      	leaq	-0x1d0(%rbp), %r14
 4faa165:      	movq	%r14, %rdi
 4faa168:      	movq	%rdx, %r15
 4faa16b:      	callq	*0x196bc7(%rip)         # 0x5140d38 <writev+0x5140d38>
 4faa171:      	movb	$0x0, -0x1d0(%rbp,%r15)
 4faa17a:      	incq	%r15
 4faa17d:      	leaq	-0x50(%rbp), %rdi
 4faa181:      	movq	%r14, %rsi
 4faa184:      	movq	%r15, %rdx
 4faa187:      	callq	*0x1a1bbb(%rip)         # 0x514bd48 <writev+0x514bd48>
 4faa18d:      	cmpl	$0x1, -0x50(%rbp)
 4faa191:      	jne	0x4faa1b2 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0x72>
 4faa193:      	leaq	0x193a16(%rip), %rax    # 0x513dbb0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x578>
 4faa19a:      	movq	%rax, -0x30(%rbp)
 4faa19e:      	movq	$-0x2, -0x38(%rbp)
 4faa1a6:      	movq	-0x38(%rbp), %r15
 4faa1aa:      	cmpq	$-0x2, %r15
 4faa1ae:      	je	0x4faa1c9 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0x89>
 4faa1b0:      	jmp	0x4faa1e7 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0xa7>
 4faa1b2:      	movq	-0x48(%rbp), %rdx
 4faa1b6:      	leaq	-0x38(%rbp), %rdi
 4faa1ba:      	callq	0x4f8e780 <_RNCNvNtNtNtCs2AWtUsOyxgP_3std3sys3env4unix6getenv0B9_>
 4faa1bf:      	movq	-0x38(%rbp), %r15
 4faa1c3:      	cmpq	$-0x2, %r15
 4faa1c7:      	jne	0x4faa1e7 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0xa7>
 4faa1c9:      	leaq	-0x30(%rbp), %rdi
 4faa1cd:      	callq	0x4f81540 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorEBH_>
 4faa1d2:      	jmp	0x4faa21e <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0xde>
 4faa1d4:      	leaq	-0x38(%rbp), %rdi
 4faa1d8:      	callq	0x4f85230 <_RINvNtNtNtCs2AWtUsOyxgP_3std3sys7helpers14small_c_string24run_with_cstr_allocatingINtNtCs4NRVxsYgnAr_4core6option6OptionNtNtNtB8_3ffi6os_str8OsStringEEB8_>
 4faa1dd:      	movq	-0x38(%rbp), %r15
 4faa1e1:      	cmpq	$-0x2, %r15
 4faa1e5:      	je	0x4faa1c9 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0x89>
 4faa1e7:      	cmpq	$-0x1, %r15
 4faa1eb:      	je	0x4faa21e <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0xde>
 4faa1ed:      	movq	-0x30(%rbp), %r12
 4faa1f1:      	movq	-0x28(%rbp), %r14
 4faa1f5:      	leaq	-0x1d0(%rbp), %rdi
 4faa1fc:      	movq	%r12, %rsi
 4faa1ff:      	movq	%r14, %rdx
 4faa202:      	callq	*0x196bb8(%rip)         # 0x5140dc0 <writev+0x5140dc0>
 4faa208:      	movq	-0x1d0(%rbp), %rax
 4faa20f:      	movq	%r15, 0x8(%rbx)
 4faa213:      	movq	%r12, 0x10(%rbx)
 4faa217:      	movl	$0x18, %ecx
 4faa21c:      	jmp	0x4faa22f <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0xef>
 4faa21e:      	movl	$0x1, %eax
 4faa223:      	movq	$-0x1, %r14
 4faa22a:      	movl	$0x8, %ecx
 4faa22f:      	movq	%r14, (%rbx,%rcx)
 4faa233:      	movq	%rax, (%rbx)
 4faa236:      	movq	%rbx, %rax
 4faa239:      	addq	$0x1b0, %rsp            # imm = 0x1B0
 4faa240:      	popq	%rbx
 4faa241:      	popq	%r12
 4faa243:      	popq	%r14
 4faa245:      	popq	%r15
 4faa247:      	popq	%rbp
 4faa248:      	retq
 4faa249:      	movq	%rax, %rbx
 4faa24c:      	testq	%r15, %r15
 4faa24f:      	je	0x4faa262 <_RNvNtCs2AWtUsOyxgP_3std3env4__var+0x122>
 4faa251:      	movl	$0x1, %edx
 4faa256:      	movq	%r12, %rdi
 4faa259:      	movq	%r15, %rsi
 4faa25c:      	callq	*0x196ade(%rip)         # 0x5140d40 <writev+0x5140d40>
 4faa262:      	movq	%rbx, %rdi
 4faa265:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 4faa26a:      	int3
 4faa26b:      	int3
 4faa26c:      	int3
 4faa26d:      	int3
 4faa26e:      	int3
 4faa26f:      	int3

0000000004fab160 <_RNvNtCs2AWtUsOyxgP_3std5alloc8rust_oom>:
 4fab160:      	pushq	%rbp
 4fab161:      	movq	%rsp, %rbp
 4fab164:      	subq	$0x10, %rsp
 4fab168:      	movq	%rdi, -0x10(%rbp)
 4fab16c:      	movq	%rsi, -0x8(%rbp)
 4fab170:      	leaq	-0x10(%rbp), %rdi
 4fab174:      	callq	*0x1a0c66(%rip)         # 0x514bde0 <writev+0x514bde0>
 4fab17a:      	int3
 4fab17b:      	int3
 4fab17c:      	int3
 4fab17d:      	int3
 4fab17e:      	int3
 4fab17f:      	int3

0000000004fab2f0 <_RNvNtCs2AWtUsOyxgP_3std7process5abort>:
 4fab2f0:      	pushq	%rbp
 4fab2f1:      	movq	%rsp, %rbp
 4fab2f4:      	callq	0x4faff60 <_RNvNtNtNtCs2AWtUsOyxgP_3std3sys3pal4unix14abort_internal>

0000000004fad030 <_RNvNtNtCs2AWtUsOyxgP_3std6thread5local18panic_access_error>:
 4fad030:      	pushq	%rbp
 4fad031:      	movq	%rsp, %rbp
 4fad034:      	subq	$0x20, %rsp
 4fad038:      	movq	%rdi, %rdx
 4fad03b:      	leaq	-0x1(%rbp), %rax
 4fad03f:      	movq	%rax, -0x18(%rbp)
 4fad043:      	movq	0x194e4e(%rip), %rax    # 0x5141e98 <writev+0x5141e98>
 4fad04a:      	movq	%rax, -0x10(%rbp)
 4fad04e:      	leaq	-0x4bdcbe9(%rip), %rdi  # 0x3d046c <anon.f4568be7443aebc71d53df3c2cdc86c9.15.llvm.6149804887664172415+0x83>
 4fad055:      	leaq	-0x18(%rbp), %rsi
 4fad059:      	callq	*0x193e21(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fad05f:      	int3

0000000004faec30 <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep>:
 4faec30:      	pushq	%rbp
 4faec31:      	movq	%rsp, %rbp
 4faec34:      	pushq	%r15
 4faec36:      	pushq	%r14
 4faec38:      	pushq	%r12
 4faec3a:      	pushq	%rbx
 4faec3b:      	subq	$0x20, %rsp
 4faec3f:      	testq	%rdi, %rdi
 4faec42:      	sete	%al
 4faec45:      	testl	%esi, %esi
 4faec47:      	sete	%cl
 4faec4a:      	testb	%cl, %al
 4faec4c:      	jne	0x4faecbd <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep+0x8d>
 4faec4e:      	movq	%rdi, %rbx
 4faec51:      	movl	%esi, %eax
 4faec53:      	movabsq	$0x7fffffffffffffff, %r15 # imm = 0x7FFFFFFFFFFFFFFF
 4faec5d:      	leaq	-0x38(%rbp), %r14
 4faec61:      	movq	0x19d590(%rip), %r12    # 0x514c1f8 <writev+0x514c1f8>
 4faec68:      	jmp	0x4faec82 <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep+0x52>
 4faec6a:      	nopw	(%rax,%rax)
 4faec70:      	movl	%eax, %eax
 4faec72:      	testq	%rbx, %rbx
 4faec75:      	setne	%cl
 4faec78:      	testq	%rax, %rax
 4faec7b:      	setg	%dl
 4faec7e:      	orb	%cl, %dl
 4faec80:      	je	0x4faecbd <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep+0x8d>
 4faec82:      	cmpq	%r15, %rbx
 4faec85:      	movq	%r15, %rcx
 4faec88:      	cmovbq	%rbx, %rcx
 4faec8c:      	movq	%rcx, -0x38(%rbp)
 4faec90:      	movq	%rax, -0x30(%rbp)
 4faec94:      	subq	%rcx, %rbx
 4faec97:      	movl	$0x1, %edi
 4faec9c:      	xorl	%esi, %esi
 4faec9e:      	movq	%r14, %rdx
 4faeca1:      	movq	%r14, %rcx
 4faeca4:      	callq	*%r12
 4faeca7:      	movl	%eax, -0x24(%rbp)
 4faecaa:      	testl	%eax, %eax
 4faecac:      	je	0x4faec70 <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep+0x40>
 4faecae:      	cmpl	$0x4, %eax
 4faecb1:      	jne	0x4faecca <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep+0x9a>
 4faecb3:      	addq	-0x38(%rbp), %rbx
 4faecb7:      	movq	-0x30(%rbp), %rax
 4faecbb:      	jmp	0x4faec72 <_RNvNtNtCs2AWtUsOyxgP_3std6thread9functions5sleep+0x42>
 4faecbd:      	addq	$0x20, %rsp
 4faecc1:      	popq	%rbx
 4faecc2:      	popq	%r12
 4faecc4:      	popq	%r14
 4faecc6:      	popq	%r15
 4faecc8:      	popq	%rbp
 4faecc9:      	retq
 4faecca:      	leaq	-0x4c21bb9(%rip), %rdx  # 0x38d118 <anon.efde23dd246e0a46063af1bb9b27e0d6.97.llvm.957717215053664604+0x8>
 4faecd1:      	leaq	0x18fcb0(%rip), %r9     # 0x513e988 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1350>
 4faecd8:      	leaq	-0x24(%rbp), %rsi
 4faecdc:      	xorl	%edi, %edi
 4faecde:      	xorl	%ecx, %ecx
 4faece0:      	callq	*0x19489a(%rip)         # 0x5143580 <writev+0x5143580>
 4faece6:      	int3
 4faece7:      	int3
 4faece8:      	int3
 4faece9:      	int3
 4faecea:      	int3
 4faeceb:      	int3
 4faecec:      	int3
 4faeced:      	int3
 4faecee:      	int3
 4faecef:      	int3

0000000004faefc0 <_RNvNtNtCs2AWtUsOyxgP_3std9panicking11panic_count17is_zero_slow_path>:
 4faefc0:      	pushq	%rbp
 4faefc1:      	movq	%rsp, %rbp
 4faefc4:      	movq	%fs:0x0, %rax
 4faefd0:      	cmpq	$0x0, -0x48(%rax)
 4faefd8:      	sete	%al
 4faefdb:      	popq	%rbp
 4faefdc:      	retq
 4faefdd:      	int3
 4faefde:      	int3
 4faefdf:      	int3

0000000004faff70 <_RNvNtNtNtCs2AWtUsOyxgP_3std3sys6random5linux19hashmap_random_keys>:
 4faff70:      	pushq	%rbp
 4faff71:      	movq	%rsp, %rbp
 4faff74:      	subq	$0x10, %rsp
 4faff78:      	xorps	%xmm0, %xmm0
 4faff7b:      	movaps	%xmm0, -0x10(%rbp)
 4faff7f:      	leaq	-0x10(%rbp), %rdi
 4faff83:      	movl	$0x10, %esi
 4faff88:      	movl	$0x1, %edx
 4faff8d:      	callq	0x4faffa0 <_RNvNtNtNtCs2AWtUsOyxgP_3std3sys6random5linux9getrandom>
 4faff92:      	movq	-0x10(%rbp), %rax
 4faff96:      	movq	-0x8(%rbp), %rdx
 4faff9a:      	addq	$0x10, %rsp
 4faff9e:      	popq	%rbp
 4faff9f:      	retq

0000000004fb54f1 <_RNvNvNtCs2AWtUsOyxgP_3std9panicking12catch_unwind7cleanup>:
 4fb54f1:      	pushq	%rbp
 4fb54f2:      	movq	%rsp, %rbp
 4fb54f5:      	pushq	%r14
 4fb54f7:      	pushq	%rbx
 4fb54f8:      	callq	*0x18b822(%rip)         # 0x5140d20 <writev+0x5140d20>
 4fb54fe:      	movq	%rax, %rbx
 4fb5501:      	movq	%rdx, %r14
 4fb5504:      	movq	0x18bb5d(%rip), %rax    # 0x5141068 <writev+0x5141068>
 4fb550b:      	lock
 4fb550c:      	decq	(%rax)
 4fb550f:      	movq	%fs:0x0, %rax
 4fb551b:      	decq	-0x48(%rax)
 4fb5522:      	movb	$0x0, -0x40(%rax)
 4fb5529:      	movq	%rbx, %rax
 4fb552c:      	movq	%r14, %rdx
 4fb552f:      	popq	%rbx
 4fb5530:      	popq	%r14
 4fb5532:      	popq	%rbp
 4fb5533:      	retq
 4fb5534:      	int3
 4fb5535:      	int3
 4fb5536:      	int3
 4fb5537:      	int3
 4fb5538:      	int3
 4fb5539:      	int3
 4fb553a:      	int3
 4fb553b:      	int3
 4fb553c:      	int3
 4fb553d:      	int3
 4fb553e:      	int3
 4fb553f:      	int3

0000000004fb57a0 <_RNvXNtNtCs2AWtUsOyxgP_3std2io5errorNtB2_5ErrorNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt>:
 4fb57a0:      	jmpq	*0x196ada(%rip)         # 0x514c280 <writev+0x514c280>
 4fb57a6:      	int3
 4fb57a7:      	int3
 4fb57a8:      	int3
 4fb57a9:      	int3
 4fb57aa:      	int3
 4fb57ab:      	int3
 4fb57ac:      	int3
 4fb57ad:      	int3
 4fb57ae:      	int3
 4fb57af:      	int3

0000000004fb7990 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt>:
 4fb7990:      	pushq	%rbp
 4fb7991:      	movq	%rsp, %rbp
 4fb7994:      	pushq	%r15
 4fb7996:      	pushq	%r14
 4fb7998:      	pushq	%rbx
 4fb7999:      	subq	$0xc8, %rsp
 4fb79a0:      	movq	(%rdi), %rbx
 4fb79a3:      	movl	%ebx, %eax
 4fb79a5:      	andl	$0x3, %eax
 4fb79a8:      	leaq	-0x27d816b(%rip), %rcx  # 0x27df844 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x19e5>
 4fb79af:      	movslq	(%rcx,%rax,4), %rax
 4fb79b3:      	addq	%rcx, %rax
 4fb79b6:      	jmpq	*%rax
 4fb79b8:      	leaq	-0x27d68a1(%rip), %rdx  # 0x27e111e <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x32bf>
 4fb79bf:      	leaq	-0xd0(%rbp), %r14
 4fb79c6:      	movl	$0x5, %ecx
 4fb79cb:      	movq	%r14, %rdi
 4fb79ce:      	callq	*0x18b3ec(%rip)         # 0x5142dc0 <writev+0x5142dc0>
 4fb79d4:      	leaq	0x10(%rbx), %rcx
 4fb79d8:      	leaq	-0x4c29d9f(%rip), %rsi  # 0x38dc40 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2450.llvm.6417388326552730234+0x20>
 4fb79df:      	leaq	0x187372(%rip), %r8     # 0x513ed58 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1720>
 4fb79e6:      	movq	0x18b3db(%rip), %r15    # 0x5142dc8 <writev+0x5142dc8>
 4fb79ed:      	movl	$0x4, %edx
 4fb79f2:      	movq	%r14, %rdi
 4fb79f5:      	callq	*%r15
 4fb79f8:      	leaq	-0x27d68e8(%rip), %rsi  # 0x27e1117 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x32b8>
 4fb79ff:      	leaq	0x187392(%rip), %r8     # 0x513ed98 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1760>
 4fb7a06:      	movl	$0x7, %edx
 4fb7a0b:      	movq	%rax, %rdi
 4fb7a0e:      	movq	%rbx, %rcx
 4fb7a11:      	callq	*%r15
 4fb7a14:      	movq	%rax, %rdi
 4fb7a17:      	callq	*0x18b3b3(%rip)         # 0x5142dd0 <writev+0x5142dd0>
 4fb7a1d:      	jmp	0x4fb7c1f <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt+0x28f>
 4fb7a22:      	shrq	$0x20, %rbx
 4fb7a26:      	movl	%ebx, -0x20(%rbp)
 4fb7a29:      	leaq	-0x27d691b(%rip), %rdx  # 0x27e1115 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x32b6>
 4fb7a30:      	leaq	-0xe0(%rbp), %rbx
 4fb7a37:      	movl	$0x2, %ecx
 4fb7a3c:      	movq	%rbx, %rdi
 4fb7a3f:      	callq	*0x18b37b(%rip)         # 0x5142dc0 <writev+0x5142dc0>
 4fb7a45:      	leaq	-0x4c2a6ec(%rip), %rsi  # 0x38d360 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2458.llvm.6417388326552730234+0x38>
 4fb7a4c:      	leaq	0x186dcd(%rip), %r8     # 0x513e820 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x11e8>
 4fb7a53:      	movq	0x18b36e(%rip), %r14    # 0x5142dc8 <writev+0x5142dc8>
 4fb7a5a:      	leaq	-0x20(%rbp), %rcx
 4fb7a5e:      	movl	$0x4, %edx
 4fb7a63:      	movq	%rbx, %rdi
 4fb7a66:      	callq	*%r14
 4fb7a69:      	movq	%rax, %rbx
 4fb7a6c:      	movl	-0x20(%rbp), %edi
 4fb7a6f:      	callq	0x4fb1b10 <_RNvNtNtNtNtCs2AWtUsOyxgP_3std3sys2io5error4unix17decode_error_kind>
 4fb7a74:      	movb	%al, -0x19(%rbp)
 4fb7a77:      	leaq	-0x4c29e3e(%rip), %rsi  # 0x38dc40 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2450.llvm.6417388326552730234+0x20>
 4fb7a7e:      	leaq	0x1872d3(%rip), %r8     # 0x513ed58 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1720>
 4fb7a85:      	leaq	-0x19(%rbp), %rcx
 4fb7a89:      	movl	$0x4, %edx
 4fb7a8e:      	movq	%rbx, %rdi
 4fb7a91:      	callq	*%r14
 4fb7a94:      	movq	%rax, %rbx
 4fb7a97:      	movl	-0x20(%rbp), %edi
 4fb7a9a:      	xorps	%xmm0, %xmm0
 4fb7a9d:      	movaps	%xmm0, -0x60(%rbp)
 4fb7aa1:      	movaps	%xmm0, -0x70(%rbp)
 4fb7aa5:      	movaps	%xmm0, -0x80(%rbp)
 4fb7aa9:      	movaps	%xmm0, -0x90(%rbp)
 4fb7ab0:      	movaps	%xmm0, -0xa0(%rbp)
 4fb7ab7:      	movaps	%xmm0, -0xb0(%rbp)
 4fb7abe:      	movaps	%xmm0, -0xc0(%rbp)
 4fb7ac5:      	movaps	%xmm0, -0xd0(%rbp)
 4fb7acc:      	leaq	-0xd0(%rbp), %rsi
 4fb7ad3:      	movl	$0x80, %edx
 4fb7ad8:      	callq	*0x192972(%rip)         # 0x514a450 <writev+0x514a450>
 4fb7ade:      	testl	%eax, %eax
 4fb7ae0:      	js	0x4fb7c2d <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt+0x29d>
 4fb7ae6:      	leaq	-0xd0(%rbp), %r14
 4fb7aed:      	movq	%r14, %rdi
 4fb7af0:      	callq	*0x18da72(%rip)         # 0x5145568 <writev+0x5145568>
 4fb7af6:      	leaq	-0x50(%rbp), %r15
 4fb7afa:      	movq	%r15, %rdi
 4fb7afd:      	movq	%r14, %rsi
 4fb7b00:      	movq	%rax, %rdx
 4fb7b03:      	callq	*0x1893df(%rip)         # 0x5140ee8 <writev+0x5140ee8>
 4fb7b09:      	leaq	-0x38(%rbp), %r14
 4fb7b0d:      	movq	%r14, %rdi
 4fb7b10:      	movq	%r15, %rsi
 4fb7b13:      	callq	*0x18cf37(%rip)         # 0x5144a50 <writev+0x5144a50>
 4fb7b19:      	leaq	-0x27d6a09(%rip), %rsi  # 0x27e1117 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x32b8>
 4fb7b20:      	leaq	0x187251(%rip), %r8     # 0x513ed78 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1740>
 4fb7b27:      	movl	$0x7, %edx
 4fb7b2c:      	movq	%rbx, %rdi
 4fb7b2f:      	movq	%r14, %rcx
 4fb7b32:      	callq	*0x18b290(%rip)         # 0x5142dc8 <writev+0x5142dc8>
 4fb7b38:      	movq	%rax, %rdi
 4fb7b3b:      	callq	*0x18b28f(%rip)         # 0x5142dd0 <writev+0x5142dd0>
 4fb7b41:      	movq	-0x38(%rbp), %rsi
 4fb7b45:      	testq	%rsi, %rsi
 4fb7b48:      	je	0x4fb7c1f <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt+0x28f>
 4fb7b4e:      	movq	-0x30(%rbp), %rdi
 4fb7b52:      	movl	$0x1, %edx
 4fb7b57:      	movl	%eax, %ebx
 4fb7b59:      	callq	*0x1891e1(%rip)         # 0x5140d40 <writev+0x5140d40>
 4fb7b5f:      	movl	%ebx, %eax
 4fb7b61:      	jmp	0x4fb7c1f <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt+0x28f>
 4fb7b66:      	movq	%rbx, %rax
 4fb7b69:      	shrq	$0x20, %rax
 4fb7b6d:      	shrq	$0x21, %rbx
 4fb7b71:      	cmpl	$0x15, %ebx
 4fb7b74:      	movl	$0xff, %ecx
 4fb7b79:      	cmovbl	%eax, %ecx
 4fb7b7c:      	movb	%cl, -0x50(%rbp)
 4fb7b7f:      	leaq	-0x4c29f56(%rip), %rdx  # 0x38dc30 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2450.llvm.6417388326552730234+0x10>
 4fb7b86:      	leaq	-0xd0(%rbp), %rbx
 4fb7b8d:      	movl	$0x4, %ecx
 4fb7b92:      	movq	%rbx, %rdi
 4fb7b95:      	callq	*0x18a27d(%rip)         # 0x5141e18 <writev+0x5141e18>
 4fb7b9b:      	leaq	0x1871b6(%rip), %rdx    # 0x513ed58 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1720>
 4fb7ba2:      	leaq	-0x50(%rbp), %rsi
 4fb7ba6:      	movq	%rbx, %rdi
 4fb7ba9:      	callq	*0x18a271(%rip)         # 0x5141e20 <writev+0x5141e20>
 4fb7baf:      	movq	%rax, %rdi
 4fb7bb2:      	callq	*0x18a270(%rip)         # 0x5141e28 <writev+0x5141e28>
 4fb7bb8:      	jmp	0x4fb7c1f <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt+0x28f>
 4fb7bba:      	leaq	-0x1(%rbx), %rax
 4fb7bbe:      	addq	$0xf, %rbx
 4fb7bc2:      	movq	%rax, -0xd0(%rbp)
 4fb7bc9:      	subq	$0x8, %rsp
 4fb7bcd:      	leaq	0x1871e4(%rip), %r10    # 0x513edb8 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1780>
 4fb7bd4:      	leaq	-0xd0(%rbp), %r11
 4fb7bdb:      	leaq	-0x27d6ab9(%rip), %r14  # 0x27e1129 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x32ca>
 4fb7be2:      	leaq	0x18716f(%rip), %r15    # 0x513ed58 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x1720>
 4fb7be9:      	leaq	-0x27d6acd(%rip), %rax  # 0x27e1123 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x32c4>
 4fb7bf0:      	leaq	-0x4c29fb7(%rip), %rcx  # 0x38dc40 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2450.llvm.6417388326552730234+0x20>
 4fb7bf7:      	movl	$0x6, %edx
 4fb7bfc:      	movl	$0x4, %r8d
 4fb7c02:      	movq	%rsi, %rdi
 4fb7c05:      	movq	%rax, %rsi
 4fb7c08:      	movq	%rbx, %r9
 4fb7c0b:      	pushq	%r10
 4fb7c0d:      	pushq	%r11
 4fb7c0f:      	pushq	$0x5
 4fb7c11:      	pushq	%r14
 4fb7c13:      	pushq	%r15
 4fb7c15:      	callq	*0x18a1d5(%rip)         # 0x5141df0 <writev+0x5141df0>
 4fb7c1b:      	addq	$0x30, %rsp
 4fb7c1f:      	addq	$0xc8, %rsp
 4fb7c26:      	popq	%rbx
 4fb7c27:      	popq	%r14
 4fb7c29:      	popq	%r15
 4fb7c2b:      	popq	%rbp
 4fb7c2c:      	retq
 4fb7c2d:      	leaq	-0x27d71df(%rip), %rdi  # 0x27e0a55 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x2bf6>
 4fb7c34:      	leaq	0x186df5(%rip), %rdx    # 0x513ea30 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x13f8>
 4fb7c3b:      	movl	$0x25, %esi
 4fb7c40:      	callq	*0x18923a(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fb7c46:      	movq	%rax, %rbx
 4fb7c49:      	movq	-0x38(%rbp), %rsi
 4fb7c4d:      	testq	%rsi, %rsi
 4fb7c50:      	je	0x4fb7c61 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std2io5errorNtNtB5_14repr_bitpacked4ReprNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt+0x2d1>
 4fb7c52:      	movq	-0x30(%rbp), %rdi
 4fb7c56:      	movl	$0x1, %edx
 4fb7c5b:      	callq	*0x1890df(%rip)         # 0x5140d40 <writev+0x5140d40>
 4fb7c61:      	movq	%rbx, %rdi
 4fb7c64:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 4fb7c69:      	int3
 4fb7c6a:      	int3
 4fb7c6b:      	int3
 4fb7c6c:      	int3
 4fb7c6d:      	int3
 4fb7c6e:      	int3
 4fb7c6f:      	int3

0000000004fb8cf0 <_RNvXs_NtCs2AWtUsOyxgP_3std4timeNtB4_7InstantINtNtNtCs4NRVxsYgnAr_4core3ops5arith3AddNtNtBN_4time8DurationE3add>:
 4fb8cf0:      	addq	%rdx, %rdi
 4fb8cf3:      	seto	%al
 4fb8cf6:      	testq	%rdx, %rdx
 4fb8cf9:      	sets	%dl
 4fb8cfc:      	xorb	%al, %dl
 4fb8cfe:      	jne	0x4fb8d1b <_RNvXs_NtCs2AWtUsOyxgP_3std4timeNtB4_7InstantINtNtNtCs4NRVxsYgnAr_4core3ops5arith3AddNtNtBN_4time8DurationE3add+0x2b>
 4fb8d00:      	addl	%esi, %ecx
 4fb8d02:      	cmpl	$0x3b9aca00, %ecx       # imm = 0x3B9ACA00
 4fb8d08:      	jb	0x4fb8d15 <_RNvXs_NtCs2AWtUsOyxgP_3std4timeNtB4_7InstantINtNtNtCs4NRVxsYgnAr_4core3ops5arith3AddNtNtBN_4time8DurationE3add+0x25>
 4fb8d0a:      	incq	%rdi
 4fb8d0d:      	jo	0x4fb8d1b <_RNvXs_NtCs2AWtUsOyxgP_3std4timeNtB4_7InstantINtNtNtCs4NRVxsYgnAr_4core3ops5arith3AddNtNtBN_4time8DurationE3add+0x2b>
 4fb8d0f:      	addl	$0xc4653600, %ecx       # imm = 0xC4653600
 4fb8d15:      	movq	%rdi, %rax
 4fb8d18:      	movl	%ecx, %edx
 4fb8d1a:      	retq
 4fb8d1b:      	pushq	%rbp
 4fb8d1c:      	movq	%rsp, %rbp
 4fb8d1f:      	leaq	-0x27d7b88(%rip), %rdi  # 0x27e119e <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x333f>
 4fb8d26:      	movl	$0x28, %esi
 4fb8d2b:      	movq	%r8, %rdx
 4fb8d2e:      	callq	*0x1880c4(%rip)         # 0x5140df8 <writev+0x5140df8>
 4fb8d34:      	int3
 4fb8d35:      	int3
 4fb8d36:      	int3
 4fb8d37:      	int3
 4fb8d38:      	int3
 4fb8d39:      	int3
 4fb8d3a:      	int3
 4fb8d3b:      	int3
 4fb8d3c:      	int3
 4fb8d3d:      	int3
 4fb8d3e:      	int3
 4fb8d3f:      	int3

0000000004fb8e50 <_RNvXs_NtNtCs2AWtUsOyxgP_3std6thread5localNtB4_11AccessErrorNtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmt>:
 4fb8e50:      	pushq	%rbp
 4fb8e51:      	movq	%rsp, %rbp
 4fb8e54:      	pushq	%rbx
 4fb8e55:      	subq	$0x18, %rsp
 4fb8e59:      	leaq	-0x27d7c9a(%rip), %rdx  # 0x27e11c6 <anon.9b5d63a33ba347daccd1b49ba12d758e.3.llvm.3439660675412778419+0x3367>
 4fb8e60:      	leaq	-0x18(%rbp), %rbx
 4fb8e64:      	movl	$0xb, %ecx
 4fb8e69:      	movq	%rbx, %rdi
 4fb8e6c:      	callq	*0x189f4e(%rip)         # 0x5142dc0 <writev+0x5142dc0>
 4fb8e72:      	movq	%rbx, %rdi
 4fb8e75:      	callq	*0x189f55(%rip)         # 0x5142dd0 <writev+0x5142dd0>
 4fb8e7b:      	addq	$0x18, %rsp
 4fb8e7f:      	popq	%rbx
 4fb8e80:      	popq	%rbp
 4fb8e81:      	retq
 4fb8e82:      	int3
 4fb8e83:      	int3
 4fb8e84:      	int3
 4fb8e85:      	int3
 4fb8e86:      	int3
 4fb8e87:      	int3
 4fb8e88:      	int3
 4fb8e89:      	int3
 4fb8e8a:      	int3
 4fb8e8b:      	int3
 4fb8e8c:      	int3
 4fb8e8d:      	int3
 4fb8e8e:      	int3
 4fb8e8f:      	int3

0000000004fbd430 <_RNvCs9wFQrvczXsK_7___rustc20___rust_panic_cleanup>:
 4fbd430:      	pushq	%rbp
 4fbd431:      	movq	%rsp, %rbp
 4fbd434:      	pushq	%r14
 4fbd436:      	pushq	%rbx
 4fbd437:      	movabsq	$0x54535552005a4f4d, %rax # imm = 0x54535552005A4F4D
 4fbd441:      	cmpq	%rax, (%rdi)
 4fbd444:      	jne	0x4fbd476 <_RNvCs9wFQrvczXsK_7___rustc20___rust_panic_cleanup+0x46>
 4fbd446:      	leaq	-0x27dbe89(%rip), %rax  # 0x27e15c4 <_RNvNtCskBg0wKg9Eqv_12panic_unwind3imp6CANARY>
 4fbd44d:      	cmpq	%rax, 0x20(%rdi)
 4fbd451:      	jne	0x4fbd47c <_RNvCs9wFQrvczXsK_7___rustc20___rust_panic_cleanup+0x4c>
 4fbd453:      	movq	0x28(%rdi), %rbx
 4fbd457:      	movq	0x30(%rdi), %r14
 4fbd45b:      	movl	$0x38, %esi
 4fbd460:      	movl	$0x8, %edx
 4fbd465:      	callq	*0x1838d5(%rip)         # 0x5140d40 <writev+0x5140d40>
 4fbd46b:      	movq	%rbx, %rax
 4fbd46e:      	movq	%r14, %rdx
 4fbd471:      	popq	%rbx
 4fbd472:      	popq	%r14
 4fbd474:      	popq	%rbp
 4fbd475:      	retq
 4fbd476:      	callq	*0x18eeb4(%rip)         # 0x514c330 <writev+0x514c330>
 4fbd47c:      	callq	*0x183886(%rip)         # 0x5140d08 <writev+0x5140d08>
 4fbd482:      	int3
 4fbd483:      	int3
 4fbd484:      	int3
 4fbd485:      	int3
 4fbd486:      	int3
 4fbd487:      	int3
 4fbd488:      	int3
 4fbd489:      	int3
 4fbd48a:      	int3
 4fbd48b:      	int3
 4fbd48c:      	int3
 4fbd48d:      	int3
 4fbd48e:      	int3
 4fbd48f:      	int3

0000000004fc7b60 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCscdodAO9FK5_5alloc6string6StringEBF_>:
 4fc7b60:      	movq	(%rdi), %rsi
 4fc7b63:      	testq	%rsi, %rsi
 4fc7b66:      	je	0x4fc7b77 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCscdodAO9FK5_5alloc6string6StringEBF_+0x17>
 4fc7b68:      	movq	0x8(%rdi), %rdi
 4fc7b6c:      	movl	$0x1, %edx
 4fc7b71:      	jmpq	*0x1791c9(%rip)         # 0x5140d40 <writev+0x5140d40>
 4fc7b77:      	retq
 4fc7b78:      	int3
 4fc7b79:      	int3
 4fc7b7a:      	int3
 4fc7b7b:      	int3
 4fc7b7c:      	int3
 4fc7b7d:      	int3
 4fc7b7e:      	int3
 4fc7b7f:      	int3

0000000004fc7cb0 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy>:
 4fc7cb0:      	pushq	%rbp
 4fc7cb1:      	movq	%rsp, %rbp
 4fc7cb4:      	pushq	%r15
 4fc7cb6:      	pushq	%r14
 4fc7cb8:      	pushq	%r13
 4fc7cba:      	pushq	%r12
 4fc7cbc:      	pushq	%rbx
 4fc7cbd:      	subq	$0x68, %rsp
 4fc7cc1:      	movq	%rdx, %r12
 4fc7cc4:      	movq	%rdi, -0x48(%rbp)
 4fc7cc8:      	movq	%rsi, -0x58(%rbp)
 4fc7ccc:      	movq	%rdx, -0x50(%rbp)
 4fc7cd0:      	leaq	-0x78(%rbp), %rdi
 4fc7cd4:      	leaq	-0x58(%rbp), %rsi
 4fc7cd8:      	callq	*0x18383a(%rip)         # 0x514b518 <writev+0x514b518>
 4fc7cde:      	movq	-0x78(%rbp), %r13
 4fc7ce2:      	testq	%r13, %r13
 4fc7ce5:      	je	0x4fc7d19 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x69>
 4fc7ce7:      	movq	-0x70(%rbp), %r15
 4fc7ceb:      	cmpq	$0x0, -0x60(%rbp)
 4fc7cf0:      	je	0x4fc7d22 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x72>
 4fc7cf2:      	testq	%r12, %r12
 4fc7cf5:      	je	0x4fc7d3a <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x8a>
 4fc7cf7:      	callq	*0x179093(%rip)         # 0x5140d90 <writev+0x5140d90>
 4fc7cfd:      	movl	$0x1, %esi
 4fc7d02:      	movq	%r12, %rdi
 4fc7d05:      	callq	*0x17908d(%rip)         # 0x5140d98 <writev+0x5140d98>
 4fc7d0b:      	testq	%rax, %rax
 4fc7d0e:      	je	0x4fc7eb8 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x208>
 4fc7d14:      	movq	%rax, %rbx
 4fc7d17:      	jmp	0x4fc7d3f <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x8f>
 4fc7d19:      	movl	$0x1, %r13d
 4fc7d1f:      	xorl	%r15d, %r15d
 4fc7d22:      	movq	-0x48(%rbp), %rax
 4fc7d26:      	movq	%r13, 0x8(%rax)
 4fc7d2a:      	movq	%r15, 0x10(%rax)
 4fc7d2e:      	movq	$-0x1, (%rax)
 4fc7d35:      	jmp	0x4fc7e6c <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x1bc>
 4fc7d3a:      	movl	$0x1, %ebx
 4fc7d3f:      	movq	%r12, -0x40(%rbp)
 4fc7d43:      	movq	%rbx, -0x38(%rbp)
 4fc7d47:      	movq	$0x0, -0x30(%rbp)
 4fc7d4f:      	cmpq	%r12, %r15
 4fc7d52:      	ja	0x4fc7e7b <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x1cb>
 4fc7d58:      	xorl	%r14d, %r14d
 4fc7d5b:      	testq	%r15, %r15
 4fc7d5e:      	je	0x4fc7d70 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0xc0>
 4fc7d60:      	leaq	(%rbx,%r14), %rdi
 4fc7d64:      	movq	%r13, %rsi
 4fc7d67:      	movq	%r15, %rdx
 4fc7d6a:      	callq	*0x178fc8(%rip)         # 0x5140d38 <writev+0x5140d38>
 4fc7d70:      	addq	%r15, %r14
 4fc7d73:      	movq	%r14, -0x30(%rbp)
 4fc7d77:      	subq	%r14, %r12
 4fc7d7a:      	cmpq	$0x2, %r12
 4fc7d7e:      	jbe	0x4fc7e9a <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x1ea>
 4fc7d84:      	movb	$-0x43, 0x2(%rbx,%r14)
 4fc7d8a:      	movw	$0xbfef, (%rbx,%r14)    # imm = 0xBFEF
 4fc7d91:      	addq	$0x3, %r14
 4fc7d95:      	movq	%r14, -0x30(%rbp)
 4fc7d99:      	movups	-0x58(%rbp), %xmm0
 4fc7d9d:      	movups	%xmm0, -0x88(%rbp)
 4fc7da4:      	leaq	-0x88(%rbp), %r12
 4fc7dab:      	nopl	(%rax,%rax)
 4fc7db0:      	leaq	-0x78(%rbp), %rdi
 4fc7db4:      	movq	%r12, %rsi
 4fc7db7:      	callq	*0x18375b(%rip)         # 0x514b518 <writev+0x514b518>
 4fc7dbd:      	movq	-0x78(%rbp), %r13
 4fc7dc1:      	testq	%r13, %r13
 4fc7dc4:      	je	0x4fc7e56 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x1a6>
 4fc7dca:      	movq	-0x70(%rbp), %rbx
 4fc7dce:      	movq	-0x60(%rbp), %r15
 4fc7dd2:      	movq	-0x40(%rbp), %rax
 4fc7dd6:      	subq	%r14, %rax
 4fc7dd9:      	cmpq	%rax, %rbx
 4fc7ddc:      	ja	0x4fc7e2a <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x17a>
 4fc7dde:      	testq	%rbx, %rbx
 4fc7de1:      	je	0x4fc7df6 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x146>
 4fc7de3:      	movq	-0x38(%rbp), %rdi
 4fc7de7:      	addq	%r14, %rdi
 4fc7dea:      	movq	%r13, %rsi
 4fc7ded:      	movq	%rbx, %rdx
 4fc7df0:      	callq	*0x178f42(%rip)         # 0x5140d38 <writev+0x5140d38>
 4fc7df6:      	addq	%rbx, %r14
 4fc7df9:      	movq	%r14, -0x30(%rbp)
 4fc7dfd:      	testq	%r15, %r15
 4fc7e00:      	je	0x4fc7db0 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x100>
 4fc7e02:      	movq	-0x40(%rbp), %rax
 4fc7e06:      	subq	%r14, %rax
 4fc7e09:      	cmpq	$0x2, %rax
 4fc7e0d:      	jbe	0x4fc7e3f <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x18f>
 4fc7e0f:      	movq	-0x38(%rbp), %rax
 4fc7e13:      	movb	$-0x43, 0x2(%rax,%r14)
 4fc7e19:      	movw	$0xbfef, (%rax,%r14)    # imm = 0xBFEF
 4fc7e20:      	addq	$0x3, %r14
 4fc7e24:      	movq	%r14, -0x30(%rbp)
 4fc7e28:      	jmp	0x4fc7db0 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x100>
 4fc7e2a:      	leaq	-0x40(%rbp), %rdi
 4fc7e2e:      	movq	%r14, %rsi
 4fc7e31:      	movq	%rbx, %rdx
 4fc7e34:      	callq	0x4fc7ba0 <_RINvNvMs2_NtCscdodAO9FK5_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalEBa_>
 4fc7e39:      	movq	-0x30(%rbp), %r14
 4fc7e3d:      	jmp	0x4fc7de3 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x133>
 4fc7e3f:      	movl	$0x3, %edx
 4fc7e44:      	leaq	-0x40(%rbp), %rdi
 4fc7e48:      	movq	%r14, %rsi
 4fc7e4b:      	callq	0x4fc7ba0 <_RINvNvMs2_NtCscdodAO9FK5_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalEBa_>
 4fc7e50:      	movq	-0x30(%rbp), %r14
 4fc7e54:      	jmp	0x4fc7e0f <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x15f>
 4fc7e56:      	movq	-0x30(%rbp), %rax
 4fc7e5a:      	movq	-0x48(%rbp), %rcx
 4fc7e5e:      	movq	%rax, 0x10(%rcx)
 4fc7e62:      	movq	%rcx, %rax
 4fc7e65:      	movups	-0x40(%rbp), %xmm0
 4fc7e69:      	movups	%xmm0, (%rcx)
 4fc7e6c:      	addq	$0x68, %rsp
 4fc7e70:      	popq	%rbx
 4fc7e71:      	popq	%r12
 4fc7e73:      	popq	%r13
 4fc7e75:      	popq	%r14
 4fc7e77:      	popq	%r15
 4fc7e79:      	popq	%rbp
 4fc7e7a:      	retq
 4fc7e7b:      	leaq	-0x40(%rbp), %rdi
 4fc7e7f:      	xorl	%esi, %esi
 4fc7e81:      	movq	%r15, %rdx
 4fc7e84:      	callq	0x4fc7ba0 <_RINvNvMs2_NtCscdodAO9FK5_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalEBa_>
 4fc7e89:      	movq	-0x30(%rbp), %r14
 4fc7e8d:      	movq	-0x40(%rbp), %r12
 4fc7e91:      	movq	-0x38(%rbp), %rbx
 4fc7e95:      	jmp	0x4fc7d60 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0xb0>
 4fc7e9a:      	leaq	-0x40(%rbp), %rdi
 4fc7e9e:      	movl	$0x3, %edx
 4fc7ea3:      	movq	%r14, %rsi
 4fc7ea6:      	callq	0x4fc7ba0 <_RINvNvMs2_NtCscdodAO9FK5_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalEBa_>
 4fc7eab:      	movq	-0x38(%rbp), %rbx
 4fc7eaf:      	movq	-0x30(%rbp), %r14
 4fc7eb3:      	jmp	0x4fc7d84 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0xd4>
 4fc7eb8:      	movl	$0x1, %edi
 4fc7ebd:      	movq	%r12, %rsi
 4fc7ec0:      	callq	*0x178ec2(%rip)         # 0x5140d88 <writev+0x5140d88>
 4fc7ec6:      	jmp	0x4fc7ec8 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x218>
 4fc7ec8:      	movq	%rax, %rbx
 4fc7ecb:      	movq	-0x40(%rbp), %rsi
 4fc7ecf:      	testq	%rsi, %rsi
 4fc7ed2:      	je	0x4fc7ee3 <_RNvMNtCscdodAO9FK5_5alloc6stringNtB2_6String15from_utf8_lossy+0x233>
 4fc7ed4:      	movq	-0x38(%rbp), %rdi
 4fc7ed8:      	movl	$0x1, %edx
 4fc7edd:      	callq	*0x178e5d(%rip)         # 0x5140d40 <writev+0x5140d40>
 4fc7ee3:      	movq	%rbx, %rdi
 4fc7ee6:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 4fc7eeb:      	int3
 4fc7eec:      	int3
 4fc7eed:      	int3
 4fc7eee:      	int3
 4fc7eef:      	int3

0000000004fc9609 <_RNvNtCscdodAO9FK5_5alloc5alloc18handle_alloc_error>:
 4fc9609:      	pushq	%rbp
 4fc960a:      	movq	%rsp, %rbp
 4fc960d:      	movq	%rdi, %rax
 4fc9610:      	movq	%rsi, %rdi
 4fc9613:      	movq	%rax, %rsi
 4fc9616:      	callq	*0x1776f4(%rip)         # 0x5140d10 <writev+0x5140d10>

0000000004fc961c <_RNvNtCscdodAO9FK5_5alloc7raw_vec12handle_error>:
 4fc961c:      	pushq	%rbp
 4fc961d:      	movq	%rsp, %rbp
 4fc9620:      	testq	%rdi, %rdi
 4fc9623:      	jne	0x4fc962b <_RNvNtCscdodAO9FK5_5alloc7raw_vec12handle_error+0xf>
 4fc9625:      	callq	*0x182e7d(%rip)         # 0x514c4a8 <writev+0x514c4a8>
 4fc962b:      	callq	*0x17792f(%rip)         # 0x5140f60 <writev+0x5140f60>
 4fc9631:      	int3
 4fc9632:      	int3
 4fc9633:      	int3
 4fc9634:      	int3
 4fc9635:      	int3
 4fc9636:      	int3
 4fc9637:      	int3
 4fc9638:      	int3
 4fc9639:      	int3
 4fc963a:      	int3
 4fc963b:      	int3
 4fc963c:      	int3
 4fc963d:      	int3
 4fc963e:      	int3
 4fc963f:      	int3

0000000004fc9640 <_RNvNtCscdodAO9FK5_5alloc7raw_vec17capacity_overflow>:
 4fc9640:      	pushq	%rbp
 4fc9641:      	movq	%rsp, %rbp
 4fc9644:      	leaq	-0x27e790d(%rip), %rdi  # 0x27e1d3e <_RNvNtCskBg0wKg9Eqv_12panic_unwind3imp6CANARY+0x77a>
 4fc964b:      	leaq	0x1767f6(%rip), %rdx    # 0x513fe48 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2810>
 4fc9652:      	movl	$0x23, %esi
 4fc9657:      	callq	*0x177823(%rip)         # 0x5140e80 <writev+0x5140e80>

0000000004fc9760 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner>:
 4fc9760:      	pushq	%rbp
 4fc9761:      	movq	%rsp, %rbp
 4fc9764:      	pushq	%r15
 4fc9766:      	pushq	%r14
 4fc9768:      	pushq	%r13
 4fc976a:      	pushq	%r12
 4fc976c:      	pushq	%rbx
 4fc976d:      	subq	$0x28, %rsp
 4fc9771:      	movq	%rdx, %r14
 4fc9774:      	movq	%rsi, %r15
 4fc9777:      	movq	%rdi, %rbx
 4fc977a:      	testb	$0x1, %r14b
 4fc977e:      	jne	0x4fc9813 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xb3>
 4fc9784:      	movzbl	(%r15), %edx
 4fc9788:      	testb	%dl, %dl
 4fc978a:      	je	0x4fc985a <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xfa>
 4fc9790:      	xorl	%eax, %eax
 4fc9792:      	movq	%r15, %rcx
 4fc9795:      	xorl	%r12d, %r12d
 4fc9798:      	jmp	0x4fc97b0 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x50>
 4fc979a:      	nopw	(%rax,%rax)
 4fc97a0:      	movzbl	%dl, %edx
 4fc97a3:      	addq	%rdx, %r12
 4fc97a6:      	addq	%rdx, %rcx
 4fc97a9:      	movzbl	(%rcx), %edx
 4fc97ac:      	testb	%dl, %dl
 4fc97ae:      	je	0x4fc981b <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xbb>
 4fc97b0:      	incq	%rcx
 4fc97b3:      	testb	%dl, %dl
 4fc97b5:      	jns	0x4fc97a0 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x40>
 4fc97b7:      	movl	%edx, %esi
 4fc97b9:      	negb	%sil
 4fc97bc:      	jno	0x4fc97cd <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x6d>
 4fc97be:      	movzwl	(%rcx), %edx
 4fc97c1:      	addq	%rdx, %r12
 4fc97c4:      	addq	%rdx, %rcx
 4fc97c7:      	addq	$0x2, %rcx
 4fc97cb:      	jmp	0x4fc97a9 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x49>
 4fc97cd:      	testq	%r12, %r12
 4fc97d0:      	sete	%sil
 4fc97d4:      	orb	%sil, %al
 4fc97d7:      	movl	%edx, %esi
 4fc97d9:      	shlb	$0x7, %sil
 4fc97dd:      	movl	%edx, %edi
 4fc97df:      	shlb	$0x5, %dil
 4fc97e3:      	andb	$0x40, %dil
 4fc97e7:      	orb	%sil, %dil
 4fc97ea:      	shrb	$0x5, %dil
 4fc97ee:      	movzbl	%dil, %esi
 4fc97f2:      	movl	%edx, %edi
 4fc97f4:      	shrb	%dil
 4fc97f7:      	andb	$0x2, %dil
 4fc97fb:      	movzbl	%dil, %edi
 4fc97ff:      	shrb	$0x2, %dl
 4fc9802:      	andb	$0x2, %dl
 4fc9805:      	movzbl	%dl, %edx
 4fc9808:      	addq	%rdi, %rcx
 4fc980b:      	addq	%rdx, %rcx
 4fc980e:      	addq	%rsi, %rcx
 4fc9811:      	jmp	0x4fc97a9 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x49>
 4fc9813:      	movq	%r14, %r12
 4fc9816:      	shrq	%r12
 4fc9819:      	jmp	0x4fc9834 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xd4>
 4fc981b:      	cmpq	$0x10, %r12
 4fc981f:      	setb	%cl
 4fc9822:      	testb	%cl, %al
 4fc9824:      	je	0x4fc982b <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xcb>
 4fc9826:      	xorl	%r12d, %r12d
 4fc9829:      	jmp	0x4fc9834 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xd4>
 4fc982b:      	addq	%r12, %r12
 4fc982e:      	js	0x4fc98d4 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x174>
 4fc9834:      	testq	%r12, %r12
 4fc9837:      	je	0x4fc985a <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0xfa>
 4fc9839:      	callq	*0x177551(%rip)         # 0x5140d90 <writev+0x5140d90>
 4fc983f:      	movl	$0x1, %r13d
 4fc9845:      	movl	$0x1, %esi
 4fc984a:      	movq	%r12, %rdi
 4fc984d:      	callq	*0x177545(%rip)         # 0x5140d98 <writev+0x5140d98>
 4fc9853:      	testq	%rax, %rax
 4fc9856:      	jne	0x4fc9862 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x102>
 4fc9858:      	jmp	0x4fc98d7 <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x177>
 4fc985a:      	movl	$0x1, %eax
 4fc985f:      	xorl	%r12d, %r12d
 4fc9862:      	movq	%r12, -0x48(%rbp)
 4fc9866:      	movq	%rax, -0x40(%rbp)
 4fc986a:      	movq	$0x0, -0x38(%rbp)
 4fc9872:      	leaq	0x1765ff(%rip), %rsi    # 0x513fe78 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2840>
 4fc9879:      	leaq	-0x48(%rbp), %rdi
 4fc987d:      	movq	%r15, %rdx
 4fc9880:      	movq	%r14, %rcx
 4fc9883:      	callq	*0x1780cf(%rip)         # 0x5141958 <writev+0x5141958>
 4fc9889:      	testb	%al, %al
 4fc988b:      	jne	0x4fc98ae <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x14e>
 4fc988d:      	movq	-0x38(%rbp), %rax
 4fc9891:      	movq	%rax, 0x10(%rbx)
 4fc9895:      	movups	-0x48(%rbp), %xmm0
 4fc9899:      	movups	%xmm0, (%rbx)
 4fc989c:      	movq	%rbx, %rax
 4fc989f:      	addq	$0x28, %rsp
 4fc98a3:      	popq	%rbx
 4fc98a4:      	popq	%r12
 4fc98a6:      	popq	%r13
 4fc98a8:      	popq	%r14
 4fc98aa:      	popq	%r15
 4fc98ac:      	popq	%rbp
 4fc98ad:      	retq
 4fc98ae:      	leaq	-0x27e7b66(%rip), %rdi  # 0x27e1d4f <_RNvNtCskBg0wKg9Eqv_12panic_unwind3imp6CANARY+0x78b>
 4fc98b5:      	leaq	0x176524(%rip), %rcx    # 0x513fde0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x27a8>
 4fc98bc:      	leaq	0x1765e5(%rip), %r8     # 0x513fea8 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2870>
 4fc98c3:      	leaq	-0x29(%rbp), %rdx
 4fc98c7:      	movl	$0x56, %esi
 4fc98cc:      	callq	*0x1775c6(%rip)         # 0x5140e98 <writev+0x5140e98>
 4fc98d2:      	ud2
 4fc98d4:      	xorl	%r13d, %r13d
 4fc98d7:      	movq	%r13, %rdi
 4fc98da:      	movq	%r12, %rsi
 4fc98dd:      	callq	*0x1774a5(%rip)         # 0x5140d88 <writev+0x5140d88>
 4fc98e3:      	movq	%rax, %rbx
 4fc98e6:      	movq	-0x48(%rbp), %rsi
 4fc98ea:      	testq	%rsi, %rsi
 4fc98ed:      	je	0x4fc98fe <_RNvNvNtCscdodAO9FK5_5alloc3fmt6format12format_inner+0x19e>
 4fc98ef:      	movq	-0x40(%rbp), %rdi
 4fc98f3:      	movl	$0x1, %edx
 4fc98f8:      	callq	*0x177442(%rip)         # 0x5140d40 <writev+0x5140d40>
 4fc98fe:      	movq	%rbx, %rdi
 4fc9901:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 4fc9906:      	int3
 4fc9907:      	int3
 4fc9908:      	int3
 4fc9909:      	int3
 4fc990a:      	int3
 4fc990b:      	int3
 4fc990c:      	int3
 4fc990d:      	int3
 4fc990e:      	int3
 4fc990f:      	int3

0000000004fc9bb0 <_RNvXs4_NtCscdodAO9FK5_5alloc6stringNtB5_6StringNtNtCs4NRVxsYgnAr_4core5clone5Clone5clone>:
 4fc9bb0:      	pushq	%rbp
 4fc9bb1:      	movq	%rsp, %rbp
 4fc9bb4:      	pushq	%r15
 4fc9bb6:      	pushq	%r14
 4fc9bb8:      	pushq	%r12
 4fc9bba:      	pushq	%rbx
 4fc9bbb:      	movq	%rdi, %r14
 4fc9bbe:      	movq	0x10(%rsi), %rbx
 4fc9bc2:      	testq	%rbx, %rbx
 4fc9bc5:      	je	0x4fc9bf8 <_RNvXs4_NtCscdodAO9FK5_5alloc6stringNtB5_6StringNtNtCs4NRVxsYgnAr_4core5clone5Clone5clone+0x48>
 4fc9bc7:      	movq	0x8(%rsi), %r12
 4fc9bcb:      	callq	*0x1771bf(%rip)         # 0x5140d90 <writev+0x5140d90>
 4fc9bd1:      	movl	$0x1, %esi
 4fc9bd6:      	movq	%rbx, %rdi
 4fc9bd9:      	callq	*0x1771b9(%rip)         # 0x5140d98 <writev+0x5140d98>
 4fc9bdf:      	testq	%rax, %rax
 4fc9be2:      	je	0x4fc9c15 <_RNvXs4_NtCscdodAO9FK5_5alloc6stringNtB5_6StringNtNtCs4NRVxsYgnAr_4core5clone5Clone5clone+0x65>
 4fc9be4:      	movq	%rax, %r15
 4fc9be7:      	movq	%rax, %rdi
 4fc9bea:      	movq	%r12, %rsi
 4fc9bed:      	movq	%rbx, %rdx
 4fc9bf0:      	callq	*0x177142(%rip)         # 0x5140d38 <writev+0x5140d38>
 4fc9bf6:      	jmp	0x4fc9bfe <_RNvXs4_NtCscdodAO9FK5_5alloc6stringNtB5_6StringNtNtCs4NRVxsYgnAr_4core5clone5Clone5clone+0x4e>
 4fc9bf8:      	movl	$0x1, %r15d
 4fc9bfe:      	movq	%rbx, (%r14)
 4fc9c01:      	movq	%r15, 0x8(%r14)
 4fc9c05:      	movq	%rbx, 0x10(%r14)
 4fc9c09:      	movq	%r14, %rax
 4fc9c0c:      	popq	%rbx
 4fc9c0d:      	popq	%r12
 4fc9c0f:      	popq	%r14
 4fc9c11:      	popq	%r15
 4fc9c13:      	popq	%rbp
 4fc9c14:      	retq
 4fc9c15:      	movl	$0x1, %edi
 4fc9c1a:      	movq	%rbx, %rsi
 4fc9c1d:      	callq	*0x177165(%rip)         # 0x5140d88 <writev+0x5140d88>
 4fc9c23:      	int3
 4fc9c24:      	int3
 4fc9c25:      	int3
 4fc9c26:      	int3
 4fc9c27:      	int3
 4fc9c28:      	int3
 4fc9c29:      	int3
 4fc9c2a:      	int3
 4fc9c2b:      	int3
 4fc9c2c:      	int3
 4fc9c2d:      	int3
 4fc9c2e:      	int3
 4fc9c2f:      	int3

0000000004fc9cd0 <_RNvXsP_NtCscdodAO9FK5_5alloc6stringNtB5_6StringINtNtCs4NRVxsYgnAr_4core7convert4FromINtNtB7_6borrow3CoweEE4from>:
 4fc9cd0:      	pushq	%rbp
 4fc9cd1:      	movq	%rsp, %rbp
 4fc9cd4:      	pushq	%r15
 4fc9cd6:      	pushq	%r14
 4fc9cd8:      	pushq	%r12
 4fc9cda:      	pushq	%rbx
 4fc9cdb:      	cmpq	$-0x1, (%rsi)
 4fc9cdf:      	je	0x4fc9cf1 <_RNvXsP_NtCscdodAO9FK5_5alloc6stringNtB5_6StringINtNtCs4NRVxsYgnAr_4core7convert4FromINtNtB7_6borrow3CoweEE4from+0x21>
 4fc9ce1:      	movq	0x10(%rsi), %rax
 4fc9ce5:      	movq	%rax, 0x10(%rdi)
 4fc9ce9:      	movups	(%rsi), %xmm0
 4fc9cec:      	movups	%xmm0, (%rdi)
 4fc9cef:      	jmp	0x4fc9d42 <_RNvXsP_NtCscdodAO9FK5_5alloc6stringNtB5_6StringINtNtCs4NRVxsYgnAr_4core7convert4FromINtNtB7_6borrow3CoweEE4from+0x72>
 4fc9cf1:      	movq	0x10(%rsi), %rbx
 4fc9cf5:      	testq	%rbx, %rbx
 4fc9cf8:      	je	0x4fc9d31 <_RNvXsP_NtCscdodAO9FK5_5alloc6stringNtB5_6StringINtNtCs4NRVxsYgnAr_4core7convert4FromINtNtB7_6borrow3CoweEE4from+0x61>
 4fc9cfa:      	movq	%rdi, %r12
 4fc9cfd:      	movq	0x8(%rsi), %r15
 4fc9d01:      	callq	*0x177089(%rip)         # 0x5140d90 <writev+0x5140d90>
 4fc9d07:      	movl	$0x1, %esi
 4fc9d0c:      	movq	%rbx, %rdi
 4fc9d0f:      	callq	*0x177083(%rip)         # 0x5140d98 <writev+0x5140d98>
 4fc9d15:      	testq	%rax, %rax
 4fc9d18:      	je	0x4fc9d4e <_RNvXsP_NtCscdodAO9FK5_5alloc6stringNtB5_6StringINtNtCs4NRVxsYgnAr_4core7convert4FromINtNtB7_6borrow3CoweEE4from+0x7e>
 4fc9d1a:      	movq	%rax, %r14
 4fc9d1d:      	movq	%rax, %rdi
 4fc9d20:      	movq	%r15, %rsi
 4fc9d23:      	movq	%rbx, %rdx
 4fc9d26:      	callq	*0x17700c(%rip)         # 0x5140d38 <writev+0x5140d38>
 4fc9d2c:      	movq	%r12, %rdi
 4fc9d2f:      	jmp	0x4fc9d37 <_RNvXsP_NtCscdodAO9FK5_5alloc6stringNtB5_6StringINtNtCs4NRVxsYgnAr_4core7convert4FromINtNtB7_6borrow3CoweEE4from+0x67>
 4fc9d31:      	movl	$0x1, %r14d
 4fc9d37:      	movq	%rbx, (%rdi)
 4fc9d3a:      	movq	%r14, 0x8(%rdi)
 4fc9d3e:      	movq	%rbx, 0x10(%rdi)
 4fc9d42:      	movq	%rdi, %rax
 4fc9d45:      	popq	%rbx
 4fc9d46:      	popq	%r12
 4fc9d48:      	popq	%r14
 4fc9d4a:      	popq	%r15
 4fc9d4c:      	popq	%rbp
 4fc9d4d:      	retq
 4fc9d4e:      	movl	$0x1, %edi
 4fc9d53:      	movq	%rbx, %rsi
 4fc9d56:      	callq	*0x17702c(%rip)         # 0x5140d88 <writev+0x5140d88>
 4fc9d5c:      	int3
 4fc9d5d:      	int3
 4fc9d5e:      	int3
 4fc9d5f:      	int3

0000000004fca2e3 <_RINvNtCs4NRVxsYgnAr_4core9panicking13assert_failedjjEB4_>:
 4fca2e3:      	pushq	%rbp
 4fca2e4:      	movq	%rsp, %rbp
 4fca2e7:      	subq	$0x10, %rsp
 4fca2eb:      	movq	%r9, %rax
 4fca2ee:      	movq	%r8, %r10
 4fca2f1:      	movq	%rcx, %r9
 4fca2f4:      	leaq	-0x8(%rbp), %r8
 4fca2f8:      	movq	%rsi, (%r8)
 4fca2fb:      	leaq	-0x10(%rbp), %rcx
 4fca2ff:      	movq	%rdx, (%rcx)
 4fca302:      	leaq	0x175c4f(%rip), %rdx    # 0x513ff58 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2920>
 4fca309:      	movq	%r8, %rsi
 4fca30c:      	movq	%rdx, %r8
 4fca30f:      	pushq	%rax
 4fca310:      	pushq	%r10
 4fca312:      	callq	*0x179eb0(%rip)         # 0x51441c8 <writev+0x51441c8>

0000000004fcd750 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive>:
 4fcd750:      	pushq	%rbp
 4fcd751:      	movq	%rsp, %rbp
 4fcd754:      	pushq	%r14
 4fcd756:      	pushq	%rbx
 4fcd757:      	subq	$0x20, %rsp
 4fcd75b:      	movq	%rdi, %rbx
 4fcd75e:      	movb	$0x1, %al
 4fcd760:      	cmpb	$0x0, 0x8(%rdi)
 4fcd764:      	jne	0x4fcd7a1 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x51>
 4fcd766:      	movq	(%rbx), %r14
 4fcd769:      	cmpb	$0x0, 0x9(%rbx)
 4fcd76d:      	je	0x4fcd78b <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x3b>
 4fcd76f:      	testb	$-0x80, 0x12(%r14)
 4fcd774:      	jne	0x4fcd7ad <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x5d>
 4fcd776:      	movq	(%r14), %rdi
 4fcd779:      	movq	0x8(%r14), %rax
 4fcd77d:      	leaq	-0x27ea442(%rip), %rsi  # 0x27e3342 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x376>
 4fcd784:      	movl	$0x6, %edx
 4fcd789:      	jmp	0x4fcd79e <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x4e>
 4fcd78b:      	movq	(%r14), %rdi
 4fcd78e:      	movq	0x8(%r14), %rax
 4fcd792:      	leaq	-0x27ea45e(%rip), %rsi  # 0x27e333b <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x36f>
 4fcd799:      	movl	$0x7, %edx
 4fcd79e:      	callq	*0x18(%rax)
 4fcd7a1:      	movb	%al, 0x8(%rbx)
 4fcd7a4:      	addq	$0x20, %rsp
 4fcd7a8:      	popq	%rbx
 4fcd7a9:      	popq	%r14
 4fcd7ab:      	popq	%rbp
 4fcd7ac:      	retq
 4fcd7ad:      	movb	$0x1, -0x11(%rbp)
 4fcd7b1:      	movups	(%r14), %xmm0
 4fcd7b5:      	movaps	%xmm0, -0x30(%rbp)
 4fcd7b9:      	leaq	-0x11(%rbp), %rax
 4fcd7bd:      	movq	%rax, -0x20(%rbp)
 4fcd7c1:      	leaq	-0x27ea480(%rip), %rsi  # 0x27e3348 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x37c>
 4fcd7c8:      	leaq	-0x30(%rbp), %rdi
 4fcd7cc:      	movl	$0x3, %edx
 4fcd7d1:      	callq	*0x17ed11(%rip)         # 0x514c4e8 <writev+0x514c4e8>
 4fcd7d7:      	testb	%al, %al
 4fcd7d9:      	je	0x4fcd7df <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x8f>
 4fcd7db:      	movb	$0x1, %al
 4fcd7dd:      	jmp	0x4fcd7a1 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x51>
 4fcd7df:      	movq	(%r14), %rdi
 4fcd7e2:      	movq	0x8(%r14), %rax
 4fcd7e6:      	leaq	-0x27ea4a2(%rip), %rsi  # 0x27e334b <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x37f>
 4fcd7ed:      	movl	$0x1, %edx
 4fcd7f2:      	jmp	0x4fcd79e <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct21finish_non_exhaustive+0x4e>
 4fcd7f4:      	int3
 4fcd7f5:      	int3
 4fcd7f6:      	int3
 4fcd7f7:      	int3
 4fcd7f8:      	int3
 4fcd7f9:      	int3
 4fcd7fa:      	int3
 4fcd7fb:      	int3
 4fcd7fc:      	int3
 4fcd7fd:      	int3
 4fcd7fe:      	int3
 4fcd7ff:      	int3

0000000004fcd800 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field>:
 4fcd800:      	pushq	%rbp
 4fcd801:      	movq	%rsp, %rbp
 4fcd804:      	pushq	%r15
 4fcd806:      	pushq	%r14
 4fcd808:      	pushq	%r13
 4fcd80a:      	pushq	%r12
 4fcd80c:      	pushq	%rbx
 4fcd80d:      	subq	$0x48, %rsp
 4fcd811:      	movq	%rdi, %rbx
 4fcd814:      	movb	$0x1, %r12b
 4fcd817:      	cmpb	$0x0, 0x8(%rdi)
 4fcd81b:      	je	0x4fcd837 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x37>
 4fcd81d:      	movb	%r12b, 0x8(%rbx)
 4fcd821:      	movb	$0x1, 0x9(%rbx)
 4fcd825:      	movq	%rbx, %rax
 4fcd828:      	addq	$0x48, %rsp
 4fcd82c:      	popq	%rbx
 4fcd82d:      	popq	%r12
 4fcd82f:      	popq	%r13
 4fcd831:      	popq	%r14
 4fcd833:      	popq	%r15
 4fcd835:      	popq	%rbp
 4fcd836:      	retq
 4fcd837:      	movq	%rcx, -0x40(%rbp)
 4fcd83b:      	movq	%r8, -0x38(%rbp)
 4fcd83f:      	movq	(%rbx), %r15
 4fcd842:      	testb	$-0x80, 0x12(%r15)
 4fcd847:      	movzbl	0x9(%rbx), %eax
 4fcd84b:      	jne	0x4fcd8c1 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0xc1>
 4fcd84d:      	movq	%rsi, %r14
 4fcd850:      	movq	%rdx, %r13
 4fcd853:      	movzbl	%al, %edx
 4fcd856:      	xorq	$0x3, %rdx
 4fcd85a:      	leaq	-0x27ea5a7(%rip), %rcx  # 0x27e32ba <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2ee>
 4fcd861:      	leaq	-0x27ea5b1(%rip), %rsi  # 0x27e32b7 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2eb>
 4fcd868:      	testb	%al, %al
 4fcd86a:      	cmovneq	%rcx, %rsi
 4fcd86e:      	movq	(%r15), %rdi
 4fcd871:      	movq	0x8(%r15), %rax
 4fcd875:      	callq	*0x18(%rax)
 4fcd878:      	testb	%al, %al
 4fcd87a:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd87c:      	movq	(%r15), %rdi
 4fcd87f:      	movq	0x8(%r15), %rax
 4fcd883:      	movq	%r14, %rsi
 4fcd886:      	movq	%r13, %rdx
 4fcd889:      	callq	*0x18(%rax)
 4fcd88c:      	testb	%al, %al
 4fcd88e:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd890:      	movq	(%r15), %rdi
 4fcd893:      	movq	0x8(%r15), %rax
 4fcd897:      	leaq	-0x27ea5e2(%rip), %rsi  # 0x27e32bc <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f0>
 4fcd89e:      	movl	$0x2, %edx
 4fcd8a3:      	callq	*0x18(%rax)
 4fcd8a6:      	testb	%al, %al
 4fcd8a8:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd8ae:      	movq	-0x40(%rbp), %rdi
 4fcd8b2:      	movq	%r15, %rsi
 4fcd8b5:      	movq	-0x38(%rbp), %rax
 4fcd8b9:      	callq	*0x18(%rax)
 4fcd8bc:      	jmp	0x4fcd97e <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x17e>
 4fcd8c1:      	testb	%al, %al
 4fcd8c3:      	jne	0x4fcd8f5 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0xf5>
 4fcd8c5:      	movq	(%r15), %rdi
 4fcd8c8:      	movq	0x8(%r15), %rcx
 4fcd8cc:      	leaq	-0x27ea615(%rip), %rax  # 0x27e32be <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f2>
 4fcd8d3:      	movq	%rdx, %r14
 4fcd8d6:      	movl	$0x3, %edx
 4fcd8db:      	movq	%rsi, %r12
 4fcd8de:      	movq	%rax, %rsi
 4fcd8e1:      	callq	*0x18(%rcx)
 4fcd8e4:      	movq	%r12, %rsi
 4fcd8e7:      	movq	%r14, %rdx
 4fcd8ea:      	movb	$0x1, %r12b
 4fcd8ed:      	testb	%al, %al
 4fcd8ef:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd8f5:      	movb	$0x1, -0x29(%rbp)
 4fcd8f9:      	movups	(%r15), %xmm0
 4fcd8fd:      	movaps	%xmm0, -0x70(%rbp)
 4fcd901:      	leaq	-0x29(%rbp), %rax
 4fcd905:      	movq	%rax, -0x60(%rbp)
 4fcd909:      	movq	0x10(%r15), %rax
 4fcd90d:      	movq	%rax, -0x48(%rbp)
 4fcd911:      	leaq	-0x70(%rbp), %rdi
 4fcd915:      	movq	%rdi, -0x58(%rbp)
 4fcd919:      	leaq	0x172890(%rip), %rax    # 0x51401b0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2b78>
 4fcd920:      	movq	%rax, -0x50(%rbp)
 4fcd924:      	callq	*0x17ebbe(%rip)         # 0x514c4e8 <writev+0x514c4e8>
 4fcd92a:      	testb	%al, %al
 4fcd92c:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd932:      	leaq	-0x27ea67d(%rip), %rsi  # 0x27e32bc <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f0>
 4fcd939:      	leaq	-0x70(%rbp), %rdi
 4fcd93d:      	movl	$0x2, %edx
 4fcd942:      	callq	*0x17eba0(%rip)         # 0x514c4e8 <writev+0x514c4e8>
 4fcd948:      	testb	%al, %al
 4fcd94a:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd950:      	leaq	-0x58(%rbp), %rsi
 4fcd954:      	movq	-0x40(%rbp), %rdi
 4fcd958:      	movq	-0x38(%rbp), %rax
 4fcd95c:      	callq	*0x18(%rax)
 4fcd95f:      	testb	%al, %al
 4fcd961:      	jne	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd967:      	movq	-0x58(%rbp), %rdi
 4fcd96b:      	movq	-0x50(%rbp), %rax
 4fcd96f:      	leaq	-0x27ea6b5(%rip), %rsi  # 0x27e32c1 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f5>
 4fcd976:      	movl	$0x2, %edx
 4fcd97b:      	callq	*0x18(%rax)
 4fcd97e:      	movl	%eax, %r12d
 4fcd981:      	jmp	0x4fcd81d <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct5field+0x1d>
 4fcd986:      	int3
 4fcd987:      	int3
 4fcd988:      	int3
 4fcd989:      	int3
 4fcd98a:      	int3
 4fcd98b:      	int3
 4fcd98c:      	int3
 4fcd98d:      	int3
 4fcd98e:      	int3
 4fcd98f:      	int3

0000000004fcd990 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct6finish>:
 4fcd990:      	movzbl	0x8(%rdi), %eax
 4fcd994:      	cmpb	$0x0, 0x9(%rdi)
 4fcd998:      	je	0x4fcd9e9 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct6finish+0x59>
 4fcd99a:      	testb	%al, %al
 4fcd99c:      	movb	$0x1, %al
 4fcd99e:      	jne	0x4fcd9e6 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct6finish+0x56>
 4fcd9a0:      	pushq	%rbp
 4fcd9a1:      	movq	%rsp, %rbp
 4fcd9a4:      	pushq	%rbx
 4fcd9a5:      	pushq	%rax
 4fcd9a6:      	movq	%rdi, %rbx
 4fcd9a9:      	movq	(%rdi), %rax
 4fcd9ac:      	testb	$-0x80, 0x12(%rax)
 4fcd9b0:      	jne	0x4fcd9c7 <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct6finish+0x37>
 4fcd9b2:      	movq	(%rax), %rdi
 4fcd9b5:      	movq	0x8(%rax), %rax
 4fcd9b9:      	leaq	-0x27ea2fd(%rip), %rsi  # 0x27e36c3 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x6f7>
 4fcd9c0:      	movl	$0x2, %edx
 4fcd9c5:      	jmp	0x4fcd9da <_RNvMs1_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_11DebugStruct6finish+0x4a>
 4fcd9c7:      	movq	(%rax), %rdi
 4fcd9ca:      	movq	0x8(%rax), %rax
 4fcd9ce:      	leaq	-0x27ea68a(%rip), %rsi  # 0x27e334b <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x37f>
 4fcd9d5:      	movl	$0x1, %edx
 4fcd9da:      	callq	*0x18(%rax)
 4fcd9dd:      	movq	%rbx, %rdi
 4fcd9e0:      	addq	$0x8, %rsp
 4fcd9e4:      	popq	%rbx
 4fcd9e5:      	popq	%rbp
 4fcd9e6:      	movb	%al, 0x8(%rdi)
 4fcd9e9:      	retq
 4fcd9ea:      	int3
 4fcd9eb:      	int3
 4fcd9ec:      	int3
 4fcd9ed:      	int3
 4fcd9ee:      	int3
 4fcd9ef:      	int3

0000000004fcd9f0 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field>:
 4fcd9f0:      	pushq	%rbp
 4fcd9f1:      	movq	%rsp, %rbp
 4fcd9f4:      	pushq	%r15
 4fcd9f6:      	pushq	%r14
 4fcd9f8:      	pushq	%r13
 4fcd9fa:      	pushq	%r12
 4fcd9fc:      	pushq	%rbx
 4fcd9fd:      	subq	$0x38, %rsp
 4fcda01:      	movq	%rdi, %rbx
 4fcda04:      	movq	(%rdi), %r15
 4fcda07:      	movb	$0x1, %al
 4fcda09:      	cmpb	$0x0, 0x10(%rdi)
 4fcda0d:      	je	0x4fcda2a <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x3a>
 4fcda0f:      	movb	%al, 0x10(%rbx)
 4fcda12:      	incq	%r15
 4fcda15:      	movq	%r15, (%rbx)
 4fcda18:      	movq	%rbx, %rax
 4fcda1b:      	addq	$0x38, %rsp
 4fcda1f:      	popq	%rbx
 4fcda20:      	popq	%r12
 4fcda22:      	popq	%r13
 4fcda24:      	popq	%r14
 4fcda26:      	popq	%r15
 4fcda28:      	popq	%rbp
 4fcda29:      	retq
 4fcda2a:      	movq	0x8(%rbx), %r14
 4fcda2e:      	testb	$-0x80, 0x12(%r14)
 4fcda33:      	jne	0x4fcda7d <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x8d>
 4fcda35:      	xorl	%eax, %eax
 4fcda37:      	testq	%r15, %r15
 4fcda3a:      	leaq	-0x27ea77e(%rip), %rdi  # 0x27e32c3 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f7>
 4fcda41:      	leaq	-0x27ea78e(%rip), %rcx  # 0x27e32ba <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2ee>
 4fcda48:      	cmoveq	%rdi, %rcx
 4fcda4c:      	setne	%al
 4fcda4f:      	incq	%rax
 4fcda52:      	movq	(%r14), %rdi
 4fcda55:      	movq	0x8(%r14), %r8
 4fcda59:      	movq	%rsi, %r12
 4fcda5c:      	movq	%rcx, %rsi
 4fcda5f:      	movq	%rdx, %r13
 4fcda62:      	movq	%rax, %rdx
 4fcda65:      	callq	*0x18(%r8)
 4fcda69:      	movl	%eax, %ecx
 4fcda6b:      	movb	$0x1, %al
 4fcda6d:      	testb	%cl, %cl
 4fcda6f:      	jne	0x4fcda0f <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x1f>
 4fcda71:      	movq	%r12, %rdi
 4fcda74:      	movq	%r14, %rsi
 4fcda77:      	callq	*0x18(%r13)
 4fcda7b:      	jmp	0x4fcda0f <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x1f>
 4fcda7d:      	testq	%r15, %r15
 4fcda80:      	jne	0x4fcdab3 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0xc3>
 4fcda82:      	movq	(%r14), %rdi
 4fcda85:      	movq	0x8(%r14), %rcx
 4fcda89:      	leaq	-0x27ea7cc(%rip), %rax  # 0x27e32c4 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f8>
 4fcda90:      	movq	%rdx, %r12
 4fcda93:      	movl	$0x2, %edx
 4fcda98:      	movq	%rsi, %r13
 4fcda9b:      	movq	%rax, %rsi
 4fcda9e:      	callq	*0x18(%rcx)
 4fcdaa1:      	movq	%r13, %rsi
 4fcdaa4:      	movq	%r12, %rdx
 4fcdaa7:      	movl	%eax, %ecx
 4fcdaa9:      	movb	$0x1, %al
 4fcdaab:      	testb	%cl, %cl
 4fcdaad:      	jne	0x4fcda0f <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x1f>
 4fcdab3:      	movb	$0x1, -0x29(%rbp)
 4fcdab7:      	movups	(%r14), %xmm0
 4fcdabb:      	movaps	%xmm0, -0x60(%rbp)
 4fcdabf:      	leaq	-0x29(%rbp), %rax
 4fcdac3:      	movq	%rax, -0x50(%rbp)
 4fcdac7:      	movq	0x10(%r14), %rax
 4fcdacb:      	movq	%rax, -0x38(%rbp)
 4fcdacf:      	leaq	-0x60(%rbp), %rax
 4fcdad3:      	movq	%rax, -0x48(%rbp)
 4fcdad7:      	leaq	0x1726d2(%rip), %rax    # 0x51401b0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2b78>
 4fcdade:      	movq	%rax, -0x40(%rbp)
 4fcdae2:      	leaq	-0x48(%rbp), %rax
 4fcdae6:      	movq	%rsi, %rdi
 4fcdae9:      	movq	%rax, %rsi
 4fcdaec:      	callq	*0x18(%rdx)
 4fcdaef:      	testb	%al, %al
 4fcdaf1:      	je	0x4fcdafa <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x10a>
 4fcdaf3:      	movb	$0x1, %al
 4fcdaf5:      	jmp	0x4fcda0f <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x1f>
 4fcdafa:      	movq	-0x48(%rbp), %rdi
 4fcdafe:      	movq	-0x40(%rbp), %rax
 4fcdb02:      	leaq	-0x27ea848(%rip), %rsi  # 0x27e32c1 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2f5>
 4fcdb09:      	movl	$0x2, %edx
 4fcdb0e:      	callq	*0x18(%rax)
 4fcdb11:      	jmp	0x4fcda0f <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple5field+0x1f>
 4fcdb16:      	int3
 4fcdb17:      	int3
 4fcdb18:      	int3
 4fcdb19:      	int3
 4fcdb1a:      	int3
 4fcdb1b:      	int3
 4fcdb1c:      	int3
 4fcdb1d:      	int3
 4fcdb1e:      	int3
 4fcdb1f:      	int3

0000000004fcdb20 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish>:
 4fcdb20:      	movq	(%rdi), %rdx
 4fcdb23:      	movzbl	0x10(%rdi), %ecx
 4fcdb27:      	testq	%rdx, %rdx
 4fcdb2a:      	je	0x4fcdb9e <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish+0x7e>
 4fcdb2c:      	movb	$0x1, %al
 4fcdb2e:      	testb	%cl, %cl
 4fcdb30:      	jne	0x4fcdb99 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish+0x79>
 4fcdb32:      	pushq	%rbp
 4fcdb33:      	movq	%rsp, %rbp
 4fcdb36:      	pushq	%r14
 4fcdb38:      	pushq	%rbx
 4fcdb39:      	movq	0x8(%rdi), %rbx
 4fcdb3d:      	cmpq	$0x1, %rdx
 4fcdb41:      	jne	0x4fcdb76 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish+0x56>
 4fcdb43:      	cmpb	$0x0, 0x11(%rdi)
 4fcdb47:      	je	0x4fcdb76 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish+0x56>
 4fcdb49:      	testb	$-0x80, 0x12(%rbx)
 4fcdb4d:      	jne	0x4fcdb76 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish+0x56>
 4fcdb4f:      	movq	(%rbx), %rax
 4fcdb52:      	movq	0x8(%rbx), %rcx
 4fcdb56:      	leaq	-0x27ea810(%rip), %rsi  # 0x27e334d <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x381>
 4fcdb5d:      	movl	$0x1, %edx
 4fcdb62:      	movq	%rdi, %r14
 4fcdb65:      	movq	%rax, %rdi
 4fcdb68:      	callq	*0x18(%rcx)
 4fcdb6b:      	movq	%r14, %rdi
 4fcdb6e:      	movl	%eax, %ecx
 4fcdb70:      	movb	$0x1, %al
 4fcdb72:      	testb	%cl, %cl
 4fcdb74:      	jne	0x4fcdb95 <_RNvMs2_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10DebugTuple6finish+0x75>
 4fcdb76:      	movq	(%rbx), %rax
 4fcdb79:      	movq	0x8(%rbx), %rcx
 4fcdb7d:      	leaq	-0x27ea838(%rip), %rsi  # 0x27e334c <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x380>
 4fcdb84:      	movl	$0x1, %edx
 4fcdb89:      	movq	%rdi, %rbx
 4fcdb8c:      	movq	%rax, %rdi
 4fcdb8f:      	callq	*0x18(%rcx)
 4fcdb92:      	movq	%rbx, %rdi
 4fcdb95:      	popq	%rbx
 4fcdb96:      	popq	%r14
 4fcdb98:      	popq	%rbp
 4fcdb99:      	movb	%al, 0x10(%rdi)
 4fcdb9c:      	movl	%eax, %ecx
 4fcdb9e:      	movl	%ecx, %eax
 4fcdba0:      	retq
 4fcdba1:      	int3
 4fcdba2:      	int3
 4fcdba3:      	int3
 4fcdba4:      	int3
 4fcdba5:      	int3
 4fcdba6:      	int3
 4fcdba7:      	int3
 4fcdba8:      	int3
 4fcdba9:      	int3
 4fcdbaa:      	int3
 4fcdbab:      	int3
 4fcdbac:      	int3
 4fcdbad:      	int3
 4fcdbae:      	int3
 4fcdbaf:      	int3

0000000004fce100 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul>:
 4fce100:      	pushq	%rbx
 4fce101:      	movq	%rdi, %rax
 4fce104:      	cmpq	$0xf, %rdx
 4fce108:      	ja	0x4fce137 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0x37>
 4fce10a:      	testq	%rdx, %rdx
 4fce10d:      	je	0x4fce1ce <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xce>
 4fce113:      	xorl	%edi, %edi
 4fce115:      	nopw	%cs:(%rax,%rax)
 4fce120:      	cmpb	$0x0, (%rsi,%rdi)
 4fce124:      	je	0x4fce1e3 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xe3>
 4fce12a:      	incq	%rdi
 4fce12d:      	cmpq	%rdi, %rdx
 4fce130:      	jne	0x4fce120 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0x20>
 4fce132:      	jmp	0x4fce1ce <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xce>
 4fce137:      	leaq	0x7(%rsi), %rcx
 4fce13b:      	andq	$-0x8, %rcx
 4fce13f:      	subq	%rsi, %rcx
 4fce142:      	jne	0x4fce191 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0x91>
 4fce144:      	leaq	-0x10(%rdx), %rdi
 4fce148:      	xorl	%ecx, %ecx
 4fce14a:      	movabsq	$-0x7f7f7f7f7f7f7f80, %r8 # imm = 0x8080808080808080
 4fce154:      	movabsq	$0x101010101010100, %r9 # imm = 0x101010101010100
 4fce15e:      	nop
 4fce160:      	movq	(%rsi,%rcx), %r10
 4fce164:      	movq	0x8(%rsi,%rcx), %r11
 4fce169:      	movq	%r9, %rbx
 4fce16c:      	subq	%r10, %rbx
 4fce16f:      	orq	%r10, %rbx
 4fce172:      	movq	%r9, %r10
 4fce175:      	subq	%r11, %r10
 4fce178:      	orq	%r11, %r10
 4fce17b:      	andq	%r8, %rbx
 4fce17e:      	andq	%r10, %rbx
 4fce181:      	cmpq	%r8, %rbx
 4fce184:      	jne	0x4fce1c9 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xc9>
 4fce186:      	addq	$0x10, %rcx
 4fce18a:      	cmpq	%rdi, %rcx
 4fce18d:      	jbe	0x4fce160 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0x60>
 4fce18f:      	jmp	0x4fce1c9 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xc9>
 4fce191:      	xorl	%edi, %edi
 4fce193:      	nopw	%cs:(%rax,%rax)
 4fce1a0:      	cmpb	$0x0, (%rsi,%rdi)
 4fce1a4:      	je	0x4fce1e3 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xe3>
 4fce1a6:      	incq	%rdi
 4fce1a9:      	cmpq	%rdi, %rcx
 4fce1ac:      	jne	0x4fce1a0 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xa0>
 4fce1ae:      	leaq	-0x10(%rdx), %rdi
 4fce1b2:      	cmpq	%rdi, %rcx
 4fce1b5:      	ja	0x4fce1c9 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xc9>
 4fce1b7:      	jmp	0x4fce14a <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0x4a>
 4fce1b9:      	nopl	(%rax)
 4fce1c0:      	cmpb	$0x0, (%rsi,%rcx)
 4fce1c4:      	je	0x4fce1e0 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xe0>
 4fce1c6:      	incq	%rcx
 4fce1c9:      	cmpq	%rcx, %rdx
 4fce1cc:      	jne	0x4fce1c0 <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xc0>
 4fce1ce:      	movq	$0x1, 0x8(%rax)
 4fce1d6:      	movl	$0x1, %ecx
 4fce1db:      	movq	%rcx, (%rax)
 4fce1de:      	popq	%rbx
 4fce1df:      	retq
 4fce1e0:      	movq	%rcx, %rdi
 4fce1e3:      	leaq	0x1(%rdi), %rcx
 4fce1e7:      	cmpq	%rdx, %rcx
 4fce1ea:      	jne	0x4fce1fb <_RNvMs3_NtNtCs4NRVxsYgnAr_4core3ffi5c_strNtB5_4CStr19from_bytes_with_nul+0xfb>
 4fce1ec:      	movq	%rsi, 0x8(%rax)
 4fce1f0:      	movq	%rdx, 0x10(%rax)
 4fce1f4:      	xorl	%ecx, %ecx
 4fce1f6:      	movq	%rcx, (%rax)
 4fce1f9:      	popq	%rbx
 4fce1fa:      	retq
 4fce1fb:      	movq	$0x0, 0x8(%rax)
 4fce203:      	movq	%rdi, 0x10(%rax)
 4fce207:      	movl	$0x1, %ecx
 4fce20c:      	movq	%rcx, (%rax)
 4fce20f:      	popq	%rbx
 4fce210:      	retq
 4fce211:      	int3
 4fce212:      	int3
 4fce213:      	int3
 4fce214:      	int3
 4fce215:      	int3
 4fce216:      	int3
 4fce217:      	int3
 4fce218:      	int3
 4fce219:      	int3
 4fce21a:      	int3
 4fce21b:      	int3
 4fce21c:      	int3
 4fce21d:      	int3
 4fce21e:      	int3
 4fce21f:      	int3

0000000004fce990 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift>:
 4fce990:      	pushq	%rbp
 4fce991:      	movq	%rsp, %rbp
 4fce994:      	pushq	%r15
 4fce996:      	pushq	%r14
 4fce998:      	pushq	%r12
 4fce99a:      	pushq	%rbx
 4fce99b:      	movq	(%rdi), %r8
 4fce99e:      	testq	%r8, %r8
 4fce9a1:      	je	0x4fceb2f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x19f>
 4fce9a7:      	movq	%rsi, %rcx
 4fce9aa:      	andl	$0x3f, %ecx
 4fce9ad:      	movl	%ecx, %eax
 4fce9af:      	leaq	-0x27e94a6(%rip), %rdx  # 0x27e5510 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase19BITSET_INDEX_CHUNKS+0x110>
 4fce9b6:      	movzwl	(%rdx,%rax,2), %r11d
 4fce9bb:      	movl	%r11d, %r9d
 4fce9be:      	shrl	$0xb, %r9d
 4fce9c2:      	andl	$0x7ff, %r11d           # imm = 0x7FF
 4fce9c9:      	movzwl	0x2(%rdx,%rax,2), %eax
 4fce9ce:      	andl	$0x7ff, %eax            # imm = 0x7FF
 4fce9d3:      	subq	%r11, %rax
 4fce9d6:      	leaq	-0x1(%rax), %rdx
 4fce9da:      	movl	$0x51c, %esi            # imm = 0x51C
 4fce9df:      	subq	%r11, %rsi
 4fce9e2:      	movl	$0x51b, %r10d           # imm = 0x51B
 4fce9e8:      	subq	%r11, %r10
 4fce9eb:      	leaq	-0x27e9460(%rip), %rbx  # 0x27e5592 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase19BITSET_INDEX_CHUNKS+0x192>
 4fce9f2:      	addq	%rbx, %r11
 4fce9f5:      	incq	%r11
 4fce9f8:      	xorl	%ebx, %ebx
 4fce9fa:      	nopw	(%rax,%rax)
 4fcea00:      	cmpq	%rbx, %rax
 4fcea03:      	je	0x4fcea62 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xd2>
 4fcea05:      	cmpq	%rbx, %rsi
 4fcea08:      	je	0x4fcea62 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xd2>
 4fcea0a:      	cmpq	%rbx, %r8
 4fcea0d:      	je	0x4fcea5f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xcf>
 4fcea0f:      	cmpq	$0x300, %rbx            # imm = 0x300
 4fcea16:      	je	0x4fceba9 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x219>
 4fcea1c:      	movzbl	-0x1(%r11,%rbx), %r14d
 4fcea22:      	movzbl	0x8(%rdi,%rbx), %r15d
 4fcea28:      	cmpb	%r14b, %r15b
 4fcea2b:      	jne	0x4fcea56 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xc6>
 4fcea2d:      	cmpq	%rbx, %rdx
 4fcea30:      	je	0x4fcea62 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xd2>
 4fcea32:      	cmpq	%rbx, %r10
 4fcea35:      	je	0x4fcea62 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xd2>
 4fcea37:      	leaq	0x1(%rbx), %r12
 4fcea3b:      	cmpq	%r8, %r12
 4fcea3e:      	je	0x4fcea5f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xcf>
 4fcea40:      	movzbl	(%r11,%rbx), %r14d
 4fcea45:      	movzbl	0x9(%rdi,%rbx), %r15d
 4fcea4b:      	incq	%r12
 4fcea4e:      	movq	%r12, %rbx
 4fcea51:      	cmpb	%r14b, %r15b
 4fcea54:      	je	0x4fcea00 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x70>
 4fcea56:      	cmpb	%r14b, %r15b
 4fcea59:      	sbbq	$0x0, %r9
 4fcea5d:      	jmp	0x4fcea62 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xd2>
 4fcea5f:      	decq	%r9
 4fcea62:      	decq	%r8
 4fcea65:      	leaq	(%r9,%rdi), %r11
 4fcea69:      	addq	$0x7, %r11
 4fcea6d:      	xorl	%edx, %edx
 4fcea6f:      	movabsq	$-0x3333333333333333, %r10 # imm = 0xCCCCCCCCCCCCCCCD
 4fcea79:      	jmp	0x4fcea8b <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xfb>
 4fcea7b:      	nopl	(%rax,%rax)
 4fcea80:      	movb	%al, 0x1(%r11,%r8)
 4fcea85:      	addq	$-0x1, %r8
 4fcea89:      	jae	0x4fceada <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x14a>
 4fcea8b:      	leaq	0x1(%r8), %rax
 4fcea8f:      	cmpq	$0x301, %rax            # imm = 0x301
 4fcea95:      	jae	0x4fceb7f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x1ef>
 4fcea9b:      	movzbl	0x8(%rdi,%r8), %esi
 4fceaa1:      	shlq	%cl, %rsi
 4fceaa4:      	addq	%rdx, %rsi
 4fceaa7:      	movq	%rsi, %rax
 4fceaaa:      	mulq	%r10
 4fceaad:      	leaq	(%r9,%r8), %rbx
 4fceab1:      	shrq	$0x3, %rdx
 4fceab5:      	leaq	(%rdx,%rdx), %rax
 4fceab9:      	leaq	(%rax,%rax,4), %r14
 4fceabd:      	movq	%rsi, %rax
 4fceac0:      	subq	%r14, %rax
 4fceac3:      	cmpq	$0x300, %rbx            # imm = 0x300
 4fceaca:      	jb	0x4fcea80 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xf0>
 4fceacc:      	testq	%rax, %rax
 4fceacf:      	je	0x4fcea85 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xf5>
 4fcead1:      	movb	$0x1, 0x30c(%rdi)
 4fcead8:      	jmp	0x4fcea85 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0xf5>
 4fceada:      	cmpq	$0xa, %rsi
 4fceade:      	jae	0x4fceb38 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x1a8>
 4fceae0:      	movq	(%rdi), %rax
 4fceae3:      	addq	%r9, %rax
 4fceae6:      	cmpq	$0x300, %rax            # imm = 0x300
 4fceaec:      	movl	$0x300, %ecx            # imm = 0x300
 4fceaf1:      	cmovbq	%rax, %rcx
 4fceaf5:      	movq	%rcx, (%rdi)
 4fceaf8:      	addl	%r9d, 0x308(%rdi)
 4fceaff:      	testq	%rax, %rax
 4fceb02:      	je	0x4fceb2f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x19f>
 4fceb04:      	nopw	%cs:(%rax,%rax)
 4fceb10:      	leaq	-0x1(%rcx), %rax
 4fceb14:      	cmpq	$0x300, %rcx            # imm = 0x300
 4fceb1b:      	ja	0x4fceb94 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x204>
 4fceb1d:      	cmpb	$0x0, 0x7(%rdi,%rcx)
 4fceb22:      	jne	0x4fceb2f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x19f>
 4fceb24:      	movq	%rax, (%rdi)
 4fceb27:      	movq	%rax, %rcx
 4fceb2a:      	testq	%rax, %rax
 4fceb2d:      	jne	0x4fceb10 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x180>
 4fceb2f:      	popq	%rbx
 4fceb30:      	popq	%r12
 4fceb32:      	popq	%r14
 4fceb34:      	popq	%r15
 4fceb36:      	popq	%rbp
 4fceb37:      	retq
 4fceb38:      	leaq	-0x1(%r9), %rsi
 4fceb3c:      	jmp	0x4fceb4d <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x1bd>
 4fceb3e:      	nop
 4fceb40:      	movb	%al, 0x8(%rdi,%rsi)
 4fceb44:      	decq	%rsi
 4fceb47:      	cmpq	$0xa, %rcx
 4fceb4b:      	jb	0x4fceae0 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x150>
 4fceb4d:      	movq	%rdx, %rcx
 4fceb50:      	movq	%rdx, %rax
 4fceb53:      	mulq	%r10
 4fceb56:      	shrq	$0x3, %rdx
 4fceb5a:      	leaq	(%rdx,%rdx), %rax
 4fceb5e:      	leaq	(%rax,%rax,4), %r8
 4fceb62:      	movq	%rcx, %rax
 4fceb65:      	subq	%r8, %rax
 4fceb68:      	cmpq	$0x300, %rsi            # imm = 0x300
 4fceb6f:      	jb	0x4fceb40 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x1b0>
 4fceb71:      	testq	%rax, %rax
 4fceb74:      	je	0x4fceb44 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x1b4>
 4fceb76:      	movb	$0x1, 0x30c(%rdi)
 4fceb7d:      	jmp	0x4fceb44 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq10left_shift+0x1b4>
 4fceb7f:      	leaq	0x171672(%rip), %rdx    # 0x51401f8 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2bc0>
 4fceb86:      	movl	$0x300, %esi            # imm = 0x300
 4fceb8b:      	movq	%r8, %rdi
 4fceb8e:      	callq	*0x1722e4(%rip)         # 0x5140e78 <writev+0x5140e78>
 4fceb94:      	leaq	0x171645(%rip), %rdx    # 0x51401e0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2ba8>
 4fceb9b:      	movl	$0x300, %esi            # imm = 0x300
 4fceba0:      	movq	%rax, %rdi
 4fceba3:      	callq	*0x1722cf(%rip)         # 0x5140e78 <writev+0x5140e78>
 4fceba9:      	leaq	0x171910(%rip), %rdx    # 0x51404c0 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data11conversions13UPPERCASE_LUT+0x70>
 4fcebb0:      	movl	$0x300, %edi            # imm = 0x300
 4fcebb5:      	movl	$0x300, %esi            # imm = 0x300
 4fcebba:      	callq	*0x1722b8(%rip)         # 0x5140e78 <writev+0x5140e78>

0000000004fcebc0 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift>:
 4fcebc0:      	pushq	%rbp
 4fcebc1:      	movq	%rsp, %rbp
 4fcebc4:      	pushq	%rbx
 4fcebc5:      	pushq	%rax
 4fcebc6:      	movq	%rsi, %rcx
 4fcebc9:      	andl	$0x3f, %ecx
 4fcebcc:      	movq	(%rdi), %rsi
 4fcebcf:      	leaq	-0x1(%rsi), %r8
 4fcebd3:      	xorl	%eax, %eax
 4fcebd5:      	xorl	%edx, %edx
 4fcebd7:      	nopw	(%rax,%rax)
 4fcebe0:      	cmpq	%rax, %rsi
 4fcebe3:      	je	0x4fcec2e <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x6e>
 4fcebe5:      	cmpq	$0x300, %rax            # imm = 0x300
 4fcebeb:      	je	0x4fced63 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x1a3>
 4fcebf1:      	leaq	(%rdx,%rdx,4), %rdx
 4fcebf5:      	movzbl	0x8(%rdi,%rax), %r9d
 4fcebfb:      	leaq	(%r9,%rdx,2), %rdx
 4fcebff:      	movq	%rdx, %r9
 4fcec02:      	shrq	%cl, %r9
 4fcec05:      	testq	%r9, %r9
 4fcec08:      	jne	0x4fcec47 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x87>
 4fcec0a:      	cmpq	%rax, %r8
 4fcec0d:      	je	0x4fcec2e <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x6e>
 4fcec0f:      	leaq	(%rdx,%rdx,4), %rdx
 4fcec13:      	movzbl	0x9(%rdi,%rax), %r9d
 4fcec19:      	leaq	(%r9,%rdx,2), %rdx
 4fcec1d:      	addq	$0x2, %rax
 4fcec21:      	movq	%rdx, %r9
 4fcec24:      	shrq	%cl, %r9
 4fcec27:      	testq	%r9, %r9
 4fcec2a:      	je	0x4fcebe0 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x20>
 4fcec2c:      	jmp	0x4fcec65 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0xa5>
 4fcec2e:      	testq	%rdx, %rdx
 4fcec31:      	je	0x4fced5c <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x19c>
 4fcec37:      	movq	%rdx, %rax
 4fcec3a:      	shrq	%cl, %rax
 4fcec3d:      	testq	%rax, %rax
 4fcec40:      	je	0x4fcec4c <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x8c>
 4fcec42:      	movq	%rsi, %rax
 4fcec45:      	jmp	0x4fcec65 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0xa5>
 4fcec47:      	incq	%rax
 4fcec4a:      	jmp	0x4fcec65 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0xa5>
 4fcec4c:      	movq	%rsi, %rax
 4fcec4f:      	nop
 4fcec50:      	addq	%rdx, %rdx
 4fcec53:      	leaq	(%rdx,%rdx,4), %rdx
 4fcec57:      	incq	%rax
 4fcec5a:      	movq	%rdx, %r8
 4fcec5d:      	shrq	%cl, %r8
 4fcec60:      	testq	%r8, %r8
 4fcec63:      	je	0x4fcec50 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x90>
 4fcec65:      	movl	0x308(%rdi), %r8d
 4fcec6c:      	subl	%eax, %r8d
 4fcec6f:      	incl	%r8d
 4fcec72:      	movl	%r8d, 0x308(%rdi)
 4fcec79:      	cmpl	$0xfffff801, %r8d       # imm = 0xFFFFF801
 4fcec80:      	jge	0x4fceca1 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0xe1>
 4fcec82:      	movq	$0x0, (%rdi)
 4fcec89:      	movl	$0x0, 0x308(%rdi)
 4fcec93:      	movb	$0x0, 0x30c(%rdi)
 4fcec9a:      	addq	$0x8, %rsp
 4fcec9e:      	popq	%rbx
 4fcec9f:      	popq	%rbp
 4fceca0:      	retq
 4fceca1:      	movq	$-0x1, %r9
 4fceca8:      	shlq	%cl, %r9
 4fcecab:      	notq	%r9
 4fcecae:      	movq	%rsi, %r8
 4fcecb1:      	subq	%rax, %r8
 4fcecb4:      	jbe	0x4fcecf4 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x134>
 4fcecb6:      	xorl	%r10d, %r10d
 4fcecb9:      	nopl	(%rax)
 4fcecc0:      	cmpq	$0x300, %rax            # imm = 0x300
 4fcecc6:      	jae	0x4fced7a <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x1ba>
 4fceccc:      	movq	%rdx, %r11
 4fceccf:      	shrq	%cl, %r11
 4fcecd2:      	andq	%r9, %rdx
 4fcecd5:      	leaq	(%rdx,%rdx,4), %rdx
 4fcecd9:      	movzbl	0x8(%rdi,%rax), %ebx
 4fcecde:      	incq	%rax
 4fcece1:      	leaq	(%rbx,%rdx,2), %rdx
 4fcece5:      	movb	%r11b, 0x8(%rdi,%r10)
 4fcecea:      	incq	%r10
 4fceced:      	cmpq	%rax, %rsi
 4fcecf0:      	jne	0x4fcecc0 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x100>
 4fcecf2:      	jmp	0x4fced12 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x152>
 4fcecf4:      	xorl	%r8d, %r8d
 4fcecf7:      	jmp	0x4fced12 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x152>
 4fcecf9:      	nopl	(%rax)
 4fced00:      	movb	%al, 0x8(%rdi,%r8)
 4fced05:      	incq	%r8
 4fced08:      	andq	%r9, %rdx
 4fced0b:      	addq	%rdx, %rdx
 4fced0e:      	leaq	(%rdx,%rdx,4), %rdx
 4fced12:      	testq	%rdx, %rdx
 4fced15:      	je	0x4fced33 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x173>
 4fced17:      	movq	%rdx, %rax
 4fced1a:      	shrq	%cl, %rax
 4fced1d:      	cmpq	$0x300, %r8             # imm = 0x300
 4fced24:      	jb	0x4fced00 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x140>
 4fced26:      	testb	%al, %al
 4fced28:      	je	0x4fced08 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x148>
 4fced2a:      	movb	$0x1, 0x30c(%rdi)
 4fced31:      	jmp	0x4fced08 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x148>
 4fced33:      	movq	%r8, %rax
 4fced36:      	nopw	%cs:(%rax,%rax)
 4fced40:      	movq	%rax, (%rdi)
 4fced43:      	testq	%rax, %rax
 4fced46:      	je	0x4fced5c <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x19c>
 4fced48:      	cmpq	$0x300, %r8             # imm = 0x300
 4fced4f:      	ja	0x4fced8f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x1cf>
 4fced51:      	cmpb	$0x0, 0x7(%rdi,%rax)
 4fced56:      	leaq	-0x1(%rax), %rax
 4fced5a:      	je	0x4fced40 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq11right_shift+0x180>
 4fced5c:      	addq	$0x8, %rsp
 4fced60:      	popq	%rbx
 4fced61:      	popq	%rbp
 4fced62:      	retq
 4fced63:      	leaq	0x1714a6(%rip), %rdx    # 0x5140210 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2bd8>
 4fced6a:      	movl	$0x300, %edi            # imm = 0x300
 4fced6f:      	movl	$0x300, %esi            # imm = 0x300
 4fced74:      	callq	*0x1720fe(%rip)         # 0x5140e78 <writev+0x5140e78>
 4fced7a:      	leaq	0x1714a7(%rip), %rdx    # 0x5140228 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2bf0>
 4fced81:      	movl	$0x300, %esi            # imm = 0x300
 4fced86:      	movq	%rax, %rdi
 4fced89:      	callq	*0x1720e9(%rip)         # 0x5140e78 <writev+0x5140e78>
 4fced8f:      	decq	%rax
 4fced92:      	leaq	0x171447(%rip), %rdx    # 0x51401e0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2ba8>
 4fced99:      	movl	$0x300, %esi            # imm = 0x300
 4fced9e:      	movq	%rax, %rdi
 4fceda1:      	callq	*0x1720d1(%rip)         # 0x5140e78 <writev+0x5140e78>
 4fceda7:      	int3
 4fceda8:      	int3
 4fceda9:      	int3
 4fcedaa:      	int3
 4fcedab:      	int3
 4fcedac:      	int3
 4fcedad:      	int3
 4fcedae:      	int3
 4fcedaf:      	int3

0000000004fcedb0 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round>:
 4fcedb0:      	movq	(%rdi), %rdx
 4fcedb3:      	testq	%rdx, %rdx
 4fcedb6:      	je	0x4fcedef <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x3f>
 4fcedb8:      	movslq	0x308(%rdi), %rcx
 4fcedbf:      	testq	%rcx, %rcx
 4fcedc2:      	js	0x4fcedef <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x3f>
 4fcedc4:      	movq	$-0x1, %rax
 4fcedcb:      	cmpl	$0x12, %ecx
 4fcedce:      	ja	0x4fceea8 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf8>
 4fcedd4:      	testl	%ecx, %ecx
 4fcedd6:      	je	0x4fcedf2 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x42>
 4fcedd8:      	cmpl	$0x1, %ecx
 4fceddb:      	jne	0x4fcedf6 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x46>
 4fceddd:      	xorl	%eax, %eax
 4fceddf:      	xorl	%esi, %esi
 4fcede1:      	addq	%rax, %rax
 4fcede4:      	leaq	(%rax,%rax,4), %rax
 4fcede8:      	cmpq	%rdx, %rsi
 4fcedeb:      	jb	0x4fcee60 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xb0>
 4fceded:      	jmp	0x4fcee68 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xb8>
 4fcedef:      	xorl	%eax, %eax
 4fcedf1:      	retq
 4fcedf2:      	xorl	%eax, %eax
 4fcedf4:      	jmp	0x4fcee68 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xb8>
 4fcedf6:      	movl	%ecx, %r8d
 4fcedf9:      	andl	$0x1e, %r8d
 4fcedfd:      	xorl	%eax, %eax
 4fcedff:      	xorl	%r9d, %r9d
 4fcee02:      	jmp	0x4fcee18 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x68>
 4fcee04:      	nopw	%cs:(%rax,%rax)
 4fcee10:      	incq	%r9
 4fcee13:      	cmpq	%r8, %r9
 4fcee16:      	je	0x4fcee4b <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x9b>
 4fcee18:      	movq	%r9, %rsi
 4fcee1b:      	addq	%rax, %rax
 4fcee1e:      	leaq	(%rax,%rax,4), %rax
 4fcee22:      	cmpq	%rdx, %r9
 4fcee25:      	jae	0x4fcee30 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x80>
 4fcee27:      	movzbl	0x8(%rdi,%rsi), %r9d
 4fcee2d:      	addq	%r9, %rax
 4fcee30:      	addq	%rax, %rax
 4fcee33:      	leaq	(%rax,%rax,4), %rax
 4fcee37:      	leaq	0x1(%rsi), %r9
 4fcee3b:      	cmpq	%rdx, %r9
 4fcee3e:      	jae	0x4fcee10 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x60>
 4fcee40:      	movzbl	0x9(%rdi,%rsi), %r10d
 4fcee46:      	addq	%r10, %rax
 4fcee49:      	jmp	0x4fcee10 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0x60>
 4fcee4b:      	testb	$0x1, %cl
 4fcee4e:      	je	0x4fcee68 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xb8>
 4fcee50:      	addq	$0x2, %rsi
 4fcee54:      	addq	%rax, %rax
 4fcee57:      	leaq	(%rax,%rax,4), %rax
 4fcee5b:      	cmpq	%rdx, %rsi
 4fcee5e:      	jae	0x4fcee68 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xb8>
 4fcee60:      	movzbl	0x8(%rdi,%rsi), %esi
 4fcee65:      	addq	%rsi, %rax
 4fcee68:      	cmpq	%rcx, %rdx
 4fcee6b:      	jbe	0x4fceea8 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf8>
 4fcee6d:      	movzbl	0x8(%rdi,%rcx), %esi
 4fcee72:      	cmpb	$0x5, %sil
 4fcee76:      	sete	%r8b
 4fcee7a:      	leaq	0x1(%rcx), %r9
 4fcee7e:      	cmpq	%rdx, %r9
 4fcee81:      	sete	%dl
 4fcee84:      	testb	%r8b, %dl
 4fcee87:      	je	0x4fcee9f <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xef>
 4fcee89:      	cmpb	$0x0, 0x30c(%rdi)
 4fcee90:      	jne	0x4fceea5 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf5>
 4fcee92:      	testl	%ecx, %ecx
 4fcee94:      	je	0x4fceea8 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf8>
 4fcee96:      	testb	$0x1, 0x7(%rdi,%rcx)
 4fcee9b:      	jne	0x4fceea5 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf5>
 4fcee9d:      	jmp	0x4fceea8 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf8>
 4fcee9f:      	cmpb	$0x4, %sil
 4fceea3:      	jbe	0x4fceea8 <_RNvMs_NtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seqNtB4_10DecimalSeq5round+0xf8>
 4fceea5:      	incq	%rax
 4fceea8:      	retq
 4fceea9:      	int3
 4fceeaa:      	int3
 4fceeab:      	int3
 4fceeac:      	int3
 4fceead:      	int3
 4fceeae:      	int3
 4fceeaf:      	int3

0000000004fceef0 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter11debug_tuple>:
 4fceef0:      	pushq	%rbp
 4fceef1:      	movq	%rsp, %rbp
 4fceef4:      	pushq	%r15
 4fceef6:      	pushq	%r14
 4fceef8:      	pushq	%rbx
 4fceef9:      	pushq	%rax
 4fceefa:      	movq	%rcx, %rbx
 4fceefd:      	movq	%rsi, %r14
 4fcef00:      	movq	%rdi, %r15
 4fcef03:      	movq	(%rsi), %rdi
 4fcef06:      	movq	0x8(%rsi), %rax
 4fcef0a:      	movq	%rdx, %rsi
 4fcef0d:      	movq	%rcx, %rdx
 4fcef10:      	callq	*0x18(%rax)
 4fcef13:      	testq	%rbx, %rbx
 4fcef16:      	movq	%r14, 0x8(%r15)
 4fcef1a:      	movb	%al, 0x10(%r15)
 4fcef1e:      	movq	$0x0, (%r15)
 4fcef25:      	sete	0x11(%r15)
 4fcef2a:      	movq	%r15, %rax
 4fcef2d:      	addq	$0x8, %rsp
 4fcef31:      	popq	%rbx
 4fcef32:      	popq	%r14
 4fcef34:      	popq	%r15
 4fcef36:      	popq	%rbp
 4fcef37:      	retq
 4fcef38:      	int3
 4fcef39:      	int3
 4fcef3a:      	int3
 4fcef3b:      	int3
 4fcef3c:      	int3
 4fcef3d:      	int3
 4fcef3e:      	int3
 4fcef3f:      	int3

0000000004fcef40 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12debug_struct>:
 4fcef40:      	pushq	%rbp
 4fcef41:      	movq	%rsp, %rbp
 4fcef44:      	pushq	%r14
 4fcef46:      	pushq	%rbx
 4fcef47:      	movq	%rsi, %rbx
 4fcef4a:      	movq	%rdi, %r14
 4fcef4d:      	movq	(%rsi), %rdi
 4fcef50:      	movq	0x8(%rsi), %rax
 4fcef54:      	movq	%rdx, %rsi
 4fcef57:      	movq	%rcx, %rdx
 4fcef5a:      	callq	*0x18(%rax)
 4fcef5d:      	movq	%rbx, (%r14)
 4fcef60:      	movb	%al, 0x8(%r14)
 4fcef64:      	movb	$0x0, 0x9(%r14)
 4fcef69:      	movq	%r14, %rax
 4fcef6c:      	popq	%rbx
 4fcef6d:      	popq	%r14
 4fcef6f:      	popq	%rbp
 4fcef70:      	retq
 4fcef71:      	int3
 4fcef72:      	int3
 4fcef73:      	int3
 4fcef74:      	int3
 4fcef75:      	int3
 4fcef76:      	int3
 4fcef77:      	int3
 4fcef78:      	int3
 4fcef79:      	int3
 4fcef7a:      	int3
 4fcef7b:      	int3
 4fcef7c:      	int3
 4fcef7d:      	int3
 4fcef7e:      	int3
 4fcef7f:      	int3

0000000004fcef80 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral>:
 4fcef80:      	pushq	%rbp
 4fcef81:      	movq	%rsp, %rbp
 4fcef84:      	pushq	%r15
 4fcef86:      	pushq	%r14
 4fcef88:      	pushq	%r13
 4fcef8a:      	pushq	%r12
 4fcef8c:      	pushq	%rbx
 4fcef8d:      	subq	$0x48, %rsp
 4fcef91:      	movq	%r9, %r14
 4fcef94:      	movq	%r8, -0x48(%rbp)
 4fcef98:      	movq	%rcx, %r13
 4fcef9b:      	movq	%rdi, %r12
 4fcef9e:      	movl	0x10(%rdi), %r15d
 4fcefa2:      	movl	%r15d, %eax
 4fcefa5:      	andl	$0x200000, %eax         # imm = 0x200000
 4fcefaa:      	cmpl	$0x1, %eax
 4fcefad:      	movl	$0x0, %ecx
 4fcefb2:      	sbbl	%ecx, %ecx
 4fcefb4:      	orl	$0x2b, %ecx
 4fcefb7:      	shrl	$0x15, %eax
 4fcefba:      	testl	%esi, %esi
 4fcefbc:      	movl	$0x2d, %r8d
 4fcefc2:      	cmovnel	%ecx, %r8d
 4fcefc6:      	movl	$0x1, %ebx
 4fcefcb:      	cmovneq	%rax, %rbx
 4fcefcf:      	addq	%r9, %rbx
 4fcefd2:      	testl	$0x800000, %r15d        # imm = 0x800000
 4fcefd9:      	movq	%r13, -0x50(%rbp)
 4fcefdd:      	jne	0x4fcf038 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0xb8>
 4fcefdf:      	xorl	%esi, %esi
 4fcefe1:      	movzwl	0x14(%r12), %r13d
 4fcefe7:      	cmpq	%r13, %rbx
 4fcefea:      	jb	0x4fcf2df <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x35f>
 4fceff0:      	movq	(%r12), %rbx
 4fceff4:      	movq	0x8(%r12), %r15
 4fceff9:      	movq	%rsi, %rcx
 4fceffc:      	movq	%rbx, %rdi
 4fcefff:      	movq	%r15, %rsi
 4fcf002:      	movl	%r8d, %edx
 4fcf005:      	movq	-0x50(%rbp), %r8
 4fcf009:      	callq	0x4fd7370 <_RNvNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB7_9Formatter12pad_integral12write_prefix>
 4fcf00e:      	movl	%eax, %ecx
 4fcf010:      	movb	$0x1, %al
 4fcf012:      	testb	%cl, %cl
 4fcf014:      	jne	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf01a:      	movq	0x18(%r15), %rax
 4fcf01e:      	movq	%rbx, %rdi
 4fcf021:      	movq	-0x48(%rbp), %rsi
 4fcf025:      	movq	%r14, %rdx
 4fcf028:      	addq	$0x48, %rsp
 4fcf02c:      	popq	%rbx
 4fcf02d:      	popq	%r12
 4fcf02f:      	popq	%r13
 4fcf031:      	popq	%r14
 4fcf033:      	popq	%r15
 4fcf035:      	popq	%rbp
 4fcf036:      	jmpq	*%rax
 4fcf038:      	cmpq	$0x20, %r13
 4fcf03c:      	jae	0x4fcf055 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0xd5>
 4fcf03e:      	testq	%r13, %r13
 4fcf041:      	movq	%rdx, %rsi
 4fcf044:      	je	0x4fcf082 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x102>
 4fcf046:      	cmpq	$0x4, %r13
 4fcf04a:      	jae	0x4fcf089 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x109>
 4fcf04c:      	xorl	%ecx, %ecx
 4fcf04e:      	xorl	%eax, %eax
 4fcf050:      	jmp	0x4fcf2b9 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x339>
 4fcf055:      	movq	%r12, -0x40(%rbp)
 4fcf059:      	movl	%r8d, -0x2c(%rbp)
 4fcf05d:      	movq	%r14, %r12
 4fcf060:      	movq	%rdx, %r14
 4fcf063:      	movq	%rdx, %rdi
 4fcf066:      	movq	%r13, %rsi
 4fcf069:      	callq	*0x176c39(%rip)         # 0x5145ca8 <writev+0x5145ca8>
 4fcf06f:      	movq	%r14, %rsi
 4fcf072:      	movq	%r12, %r14
 4fcf075:      	movl	-0x2c(%rbp), %r8d
 4fcf079:      	movq	-0x40(%rbp), %r12
 4fcf07d:      	jmp	0x4fcf2cd <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x34d>
 4fcf082:      	xorl	%eax, %eax
 4fcf084:      	jmp	0x4fcf2cd <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x34d>
 4fcf089:      	movl	%r13d, %ecx
 4fcf08c:      	andl	$0x1c, %ecx
 4fcf08f:      	movzwl	(%rsi), %eax
 4fcf092:      	movd	%eax, %xmm0
 4fcf096:      	movzwl	0x2(%rsi), %eax
 4fcf09a:      	movd	%eax, %xmm1
 4fcf09e:      	movdqa	-0x4c47916(%rip), %xmm2 # 0x387790 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x470>
 4fcf0a6:      	pcmpgtb	%xmm2, %xmm0
 4fcf0aa:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf0ae:      	pshuflw	$0xd4, %xmm0, %xmm0     # xmm0 = xmm0[0,1,1,3,4,5,6,7]
 4fcf0b3:      	pshufd	$0xd4, %xmm0, %xmm0     # xmm0 = xmm0[0,1,1,3]
 4fcf0b8:      	movdqa	-0x4c47920(%rip), %xmm3 # 0x3877a0 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x480>
 4fcf0c0:      	pand	%xmm3, %xmm0
 4fcf0c4:      	pcmpgtb	%xmm2, %xmm1
 4fcf0c8:      	punpcklbw	%xmm1, %xmm1    # xmm1 = xmm1[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf0cc:      	pshuflw	$0xd4, %xmm1, %xmm1     # xmm1 = xmm1[0,1,1,3,4,5,6,7]
 4fcf0d1:      	pshufd	$0xd4, %xmm1, %xmm1     # xmm1 = xmm1[0,1,1,3]
 4fcf0d6:      	pand	%xmm3, %xmm1
 4fcf0da:      	cmpq	$0x4, %rcx
 4fcf0de:      	je	0x4fcf2a5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x325>
 4fcf0e4:      	movzwl	0x4(%rsi), %eax
 4fcf0e8:      	movd	%eax, %xmm4
 4fcf0ec:      	movzwl	0x6(%rsi), %eax
 4fcf0f0:      	movd	%eax, %xmm5
 4fcf0f4:      	pcmpgtb	%xmm2, %xmm4
 4fcf0f8:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf0fc:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fcf101:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fcf106:      	pand	%xmm3, %xmm4
 4fcf10a:      	pcmpgtb	%xmm2, %xmm5
 4fcf10e:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf112:      	pshuflw	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3,4,5,6,7]
 4fcf117:      	pshufd	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3]
 4fcf11c:      	pand	%xmm3, %xmm5
 4fcf120:      	paddq	%xmm4, %xmm0
 4fcf124:      	paddq	%xmm5, %xmm1
 4fcf128:      	cmpl	$0x8, %ecx
 4fcf12b:      	je	0x4fcf2a5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x325>
 4fcf131:      	movzwl	0x8(%rsi), %eax
 4fcf135:      	movd	%eax, %xmm4
 4fcf139:      	movzwl	0xa(%rsi), %eax
 4fcf13d:      	movd	%eax, %xmm5
 4fcf141:      	pcmpgtb	%xmm2, %xmm4
 4fcf145:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf149:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fcf14e:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fcf153:      	pand	%xmm3, %xmm4
 4fcf157:      	pcmpgtb	%xmm2, %xmm5
 4fcf15b:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf15f:      	pshuflw	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3,4,5,6,7]
 4fcf164:      	pshufd	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3]
 4fcf169:      	pand	%xmm3, %xmm5
 4fcf16d:      	paddq	%xmm4, %xmm0
 4fcf171:      	paddq	%xmm5, %xmm1
 4fcf175:      	cmpl	$0xc, %ecx
 4fcf178:      	je	0x4fcf2a5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x325>
 4fcf17e:      	movzwl	0xc(%rsi), %eax
 4fcf182:      	movd	%eax, %xmm4
 4fcf186:      	movzwl	0xe(%rsi), %eax
 4fcf18a:      	movd	%eax, %xmm5
 4fcf18e:      	pcmpgtb	%xmm2, %xmm4
 4fcf192:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf196:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fcf19b:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fcf1a0:      	pand	%xmm3, %xmm4
 4fcf1a4:      	pcmpgtb	%xmm2, %xmm5
 4fcf1a8:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf1ac:      	pshuflw	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3,4,5,6,7]
 4fcf1b1:      	pshufd	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3]
 4fcf1b6:      	pand	%xmm3, %xmm5
 4fcf1ba:      	paddq	%xmm4, %xmm0
 4fcf1be:      	paddq	%xmm5, %xmm1
 4fcf1c2:      	cmpl	$0x10, %ecx
 4fcf1c5:      	je	0x4fcf2a5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x325>
 4fcf1cb:      	movzwl	0x10(%rsi), %eax
 4fcf1cf:      	movd	%eax, %xmm4
 4fcf1d3:      	movzwl	0x12(%rsi), %eax
 4fcf1d7:      	movd	%eax, %xmm5
 4fcf1db:      	pcmpgtb	%xmm2, %xmm4
 4fcf1df:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf1e3:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fcf1e8:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fcf1ed:      	pand	%xmm3, %xmm4
 4fcf1f1:      	pcmpgtb	%xmm2, %xmm5
 4fcf1f5:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf1f9:      	pshuflw	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3,4,5,6,7]
 4fcf1fe:      	pshufd	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3]
 4fcf203:      	pand	%xmm3, %xmm5
 4fcf207:      	paddq	%xmm4, %xmm0
 4fcf20b:      	paddq	%xmm5, %xmm1
 4fcf20f:      	cmpl	$0x14, %ecx
 4fcf212:      	je	0x4fcf2a5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x325>
 4fcf218:      	movzwl	0x14(%rsi), %eax
 4fcf21c:      	movd	%eax, %xmm4
 4fcf220:      	movzwl	0x16(%rsi), %eax
 4fcf224:      	movd	%eax, %xmm5
 4fcf228:      	pcmpgtb	%xmm2, %xmm4
 4fcf22c:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf230:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fcf235:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fcf23a:      	pand	%xmm3, %xmm4
 4fcf23e:      	pcmpgtb	%xmm2, %xmm5
 4fcf242:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf246:      	pshuflw	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3,4,5,6,7]
 4fcf24b:      	pshufd	$0xd4, %xmm5, %xmm5     # xmm5 = xmm5[0,1,1,3]
 4fcf250:      	pand	%xmm3, %xmm5
 4fcf254:      	paddq	%xmm4, %xmm0
 4fcf258:      	paddq	%xmm5, %xmm1
 4fcf25c:      	cmpl	$0x18, %ecx
 4fcf25f:      	je	0x4fcf2a5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x325>
 4fcf261:      	movzwl	0x18(%rsi), %eax
 4fcf265:      	movd	%eax, %xmm4
 4fcf269:      	movzwl	0x1a(%rsi), %eax
 4fcf26d:      	movd	%eax, %xmm5
 4fcf271:      	pcmpgtb	%xmm2, %xmm4
 4fcf275:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf279:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fcf27e:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fcf283:      	pand	%xmm3, %xmm4
 4fcf287:      	pcmpgtb	%xmm2, %xmm5
 4fcf28b:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf28f:      	pshuflw	$0xd4, %xmm5, %xmm2     # xmm2 = xmm5[0,1,1,3,4,5,6,7]
 4fcf294:      	pshufd	$0xd4, %xmm2, %xmm2     # xmm2 = xmm2[0,1,1,3]
 4fcf299:      	pand	%xmm3, %xmm2
 4fcf29d:      	paddq	%xmm4, %xmm0
 4fcf2a1:      	paddq	%xmm2, %xmm1
 4fcf2a5:      	paddq	%xmm1, %xmm0
 4fcf2a9:      	pshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
 4fcf2ae:      	paddq	%xmm0, %xmm1
 4fcf2b2:      	movq	%xmm1, %rax
 4fcf2b7:      	jmp	0x4fcf2c8 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x348>
 4fcf2b9:      	xorl	%edx, %edx
 4fcf2bb:      	cmpb	$-0x40, (%rsi,%rcx)
 4fcf2bf:      	setge	%dl
 4fcf2c2:      	addq	%rdx, %rax
 4fcf2c5:      	incq	%rcx
 4fcf2c8:      	cmpq	%rcx, %r13
 4fcf2cb:      	jne	0x4fcf2b9 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x339>
 4fcf2cd:      	addq	%rax, %rbx
 4fcf2d0:      	movzwl	0x14(%r12), %r13d
 4fcf2d6:      	cmpq	%r13, %rbx
 4fcf2d9:      	jae	0x4fceff0 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x70>
 4fcf2df:      	testl	$0x1000000, %r15d       # imm = 0x1000000
 4fcf2e6:      	movq	%r14, -0x58(%rbp)
 4fcf2ea:      	jne	0x4fcf320 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x3a0>
 4fcf2ec:      	movl	%r13d, %edx
 4fcf2ef:      	subl	%ebx, %edx
 4fcf2f1:      	movl	%r15d, %eax
 4fcf2f4:      	shrl	$0x1d, %eax
 4fcf2f7:      	andl	$0x3, %eax
 4fcf2fa:      	leaq	-0x27ed35d(%rip), %rcx  # 0x27e1fa4 <_RNvNtCskBg0wKg9Eqv_12panic_unwind3imp6CANARY+0x9e0>
 4fcf301:      	movslq	(%rcx,%rax,4), %rax
 4fcf305:      	addq	%rcx, %rax
 4fcf308:      	movq	%rbx, -0x68(%rbp)
 4fcf30c:      	movq	%rsi, -0x40(%rbp)
 4fcf310:      	movl	%r8d, -0x2c(%rbp)
 4fcf314:      	movl	%edx, -0x38(%rbp)
 4fcf317:      	jmpq	*%rax
 4fcf319:      	movl	%edx, %eax
 4fcf31b:      	jmp	0x4fcf3b7 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x437>
 4fcf320:      	movq	%rbx, %r14
 4fcf323:      	movq	0x10(%r12), %rax
 4fcf328:      	movq	%rax, -0x38(%rbp)
 4fcf32c:      	andl	$0x9fe00000, %eax       # imm = 0x9FE00000
 4fcf331:      	orl	$0x20000030, %eax       # imm = 0x20000030
 4fcf336:      	movl	%eax, 0x10(%r12)
 4fcf33b:      	movq	(%r12), %r15
 4fcf33f:      	movq	0x8(%r12), %rbx
 4fcf344:      	movq	%rsi, %rcx
 4fcf347:      	movq	%r15, %rdi
 4fcf34a:      	movq	%rbx, %rsi
 4fcf34d:      	movl	%r8d, %edx
 4fcf350:      	movq	-0x50(%rbp), %r8
 4fcf354:      	callq	0x4fd7370 <_RNvNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB7_9Formatter12pad_integral12write_prefix>
 4fcf359:      	movl	%eax, %ecx
 4fcf35b:      	movb	$0x1, %al
 4fcf35d:      	testb	%cl, %cl
 4fcf35f:      	jne	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf365:      	subl	%r14d, %r13d
 4fcf368:      	incl	%r13d
 4fcf36b:      	movq	-0x58(%rbp), %r14
 4fcf36f:      	nop
 4fcf370:      	decw	%r13w
 4fcf374:      	je	0x4fcf387 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x407>
 4fcf376:      	movq	%r15, %rdi
 4fcf379:      	movl	$0x30, %esi
 4fcf37e:      	callq	*0x20(%rbx)
 4fcf381:      	testb	%al, %al
 4fcf383:      	je	0x4fcf370 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x3f0>
 4fcf385:      	jmp	0x4fcf3e5 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x465>
 4fcf387:      	movq	%r15, %rdi
 4fcf38a:      	movq	-0x48(%rbp), %rsi
 4fcf38e:      	movq	%r14, %rdx
 4fcf391:      	callq	*0x18(%rbx)
 4fcf394:      	testb	%al, %al
 4fcf396:      	movb	$0x1, %al
 4fcf398:      	jne	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf39e:      	movq	-0x38(%rbp), %rax
 4fcf3a2:      	movq	%rax, 0x10(%r12)
 4fcf3a7:      	xorl	%eax, %eax
 4fcf3a9:      	jmp	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf3ae:      	xorl	%eax, %eax
 4fcf3b0:      	jmp	0x4fcf3b7 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x437>
 4fcf3b2:      	movzwl	%dx, %eax
 4fcf3b5:      	shrl	%eax
 4fcf3b7:      	andl	$0x1fffff, %r15d        # imm = 0x1FFFFF
 4fcf3be:      	movq	(%r12), %rbx
 4fcf3c2:      	movq	0x8(%r12), %r12
 4fcf3c7:      	movq	%rax, -0x60(%rbp)
 4fcf3cb:      	leal	0x1(%rax), %r14d
 4fcf3cf:      	nop
 4fcf3d0:      	movq	%rbx, %rdi
 4fcf3d3:      	decw	%r14w
 4fcf3d7:      	je	0x4fcf3e9 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x469>
 4fcf3d9:      	movl	%r15d, %esi
 4fcf3dc:      	callq	*0x20(%r12)
 4fcf3e1:      	testb	%al, %al
 4fcf3e3:      	je	0x4fcf3d0 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x450>
 4fcf3e5:      	movb	$0x1, %al
 4fcf3e7:      	jmp	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf3e9:      	movq	%r12, %rsi
 4fcf3ec:      	movl	-0x2c(%rbp), %edx
 4fcf3ef:      	movq	-0x40(%rbp), %rcx
 4fcf3f3:      	movq	-0x50(%rbp), %r8
 4fcf3f7:      	callq	0x4fd7370 <_RNvNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB7_9Formatter12pad_integral12write_prefix>
 4fcf3fc:      	movl	%eax, %ecx
 4fcf3fe:      	movb	$0x1, %al
 4fcf400:      	testb	%cl, %cl
 4fcf402:      	jne	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf404:      	movq	%rbx, %rdi
 4fcf407:      	movq	-0x48(%rbp), %rsi
 4fcf40b:      	movq	-0x58(%rbp), %rdx
 4fcf40f:      	callq	*0x18(%r12)
 4fcf414:      	movl	%eax, %ecx
 4fcf416:      	movb	$0x1, %al
 4fcf418:      	testb	%cl, %cl
 4fcf41a:      	jne	0x4fcf462 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4e2>
 4fcf41c:      	movq	-0x60(%rbp), %rcx
 4fcf420:      	subl	%ecx, -0x38(%rbp)
 4fcf423:      	addl	-0x68(%rbp), %ecx
 4fcf426:      	subl	%r13d, %ecx
 4fcf429:      	movw	$0xffff, %r14w          # imm = 0xFFFF
 4fcf42e:      	nop
 4fcf430:      	leal	(%rcx,%r14), %eax
 4fcf434:      	cmpw	$-0x1, %ax
 4fcf438:      	je	0x4fcf454 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4d4>
 4fcf43a:      	movq	%rbx, %rdi
 4fcf43d:      	movl	%r15d, %esi
 4fcf440:      	movq	%rcx, %r13
 4fcf443:      	callq	*0x20(%r12)
 4fcf448:      	movq	%r13, %rcx
 4fcf44b:      	incl	%r14d
 4fcf44e:      	testb	%al, %al
 4fcf450:      	je	0x4fcf430 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4b0>
 4fcf452:      	jmp	0x4fcf45a <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter12pad_integral+0x4da>
 4fcf454:      	movl	-0x38(%rbp), %eax
 4fcf457:      	movl	%eax, %r14d
 4fcf45a:      	cmpw	-0x38(%rbp), %r14w
 4fcf45f:      	setb	%al
 4fcf462:      	addq	$0x48, %rsp
 4fcf466:      	popq	%rbx
 4fcf467:      	popq	%r12
 4fcf469:      	popq	%r13
 4fcf46b:      	popq	%r14
 4fcf46d:      	popq	%r15
 4fcf46f:      	popq	%rbp
 4fcf470:      	retq
 4fcf471:      	int3
 4fcf472:      	int3
 4fcf473:      	int3
 4fcf474:      	int3
 4fcf475:      	int3
 4fcf476:      	int3
 4fcf477:      	int3
 4fcf478:      	int3
 4fcf479:      	int3
 4fcf47a:      	int3
 4fcf47b:      	int3
 4fcf47c:      	int3
 4fcf47d:      	int3
 4fcf47e:      	int3
 4fcf47f:      	int3

0000000004fd0120 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter26debug_struct_field2_finish>:
 4fd0120:      	pushq	%rbp
 4fd0121:      	movq	%rsp, %rbp
 4fd0124:      	pushq	%r15
 4fd0126:      	pushq	%r14
 4fd0128:      	pushq	%r13
 4fd012a:      	pushq	%r12
 4fd012c:      	pushq	%rbx
 4fd012d:      	subq	$0x18, %rsp
 4fd0131:      	movq	%r9, %rbx
 4fd0134:      	movq	%r8, %r14
 4fd0137:      	movq	%rcx, %r15
 4fd013a:      	movq	%rdi, %r12
 4fd013d:      	movq	(%rdi), %rdi
 4fd0140:      	movq	0x8(%r12), %rax
 4fd0145:      	callq	*0x18(%rax)
 4fd0148:      	movq	%r12, -0x38(%rbp)
 4fd014c:      	movb	%al, -0x30(%rbp)
 4fd014f:      	movb	$0x0, -0x2f(%rbp)
 4fd0153:      	movq	0x172c6e(%rip), %r13    # 0x5142dc8 <writev+0x5142dc8>
 4fd015a:      	leaq	-0x38(%rbp), %r12
 4fd015e:      	movq	%r12, %rdi
 4fd0161:      	movq	%r15, %rsi
 4fd0164:      	movq	%r14, %rdx
 4fd0167:      	movq	%rbx, %rcx
 4fd016a:      	movq	0x10(%rbp), %r8
 4fd016e:      	callq	*%r13
 4fd0171:      	movq	%r12, %rdi
 4fd0174:      	movq	0x18(%rbp), %rsi
 4fd0178:      	movq	0x20(%rbp), %rdx
 4fd017c:      	movq	0x28(%rbp), %rcx
 4fd0180:      	movq	0x30(%rbp), %r8
 4fd0184:      	callq	*%r13
 4fd0187:      	movzbl	-0x30(%rbp), %ecx
 4fd018b:      	movzbl	-0x2f(%rbp), %eax
 4fd018f:      	movl	%eax, %edx
 4fd0191:      	notb	%dl
 4fd0193:      	orb	%cl, %dl
 4fd0195:      	testb	$0x1, %dl
 4fd0198:      	je	0x4fd019e <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter26debug_struct_field2_finish+0x7e>
 4fd019a:      	orb	%cl, %al
 4fd019c:      	jmp	0x4fd01d3 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter26debug_struct_field2_finish+0xb3>
 4fd019e:      	movq	-0x38(%rbp), %rax
 4fd01a2:      	testb	$-0x80, 0x12(%rax)
 4fd01a6:      	jne	0x4fd01bd <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter26debug_struct_field2_finish+0x9d>
 4fd01a8:      	movq	(%rax), %rdi
 4fd01ab:      	movq	0x8(%rax), %rax
 4fd01af:      	leaq	-0x27ecaf3(%rip), %rsi  # 0x27e36c3 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x6f7>
 4fd01b6:      	movl	$0x2, %edx
 4fd01bb:      	jmp	0x4fd01d0 <_RNvMsa_NtCs4NRVxsYgnAr_4core3fmtNtB5_9Formatter26debug_struct_field2_finish+0xb0>
 4fd01bd:      	movq	(%rax), %rdi
 4fd01c0:      	movq	0x8(%rax), %rax
 4fd01c4:      	leaq	-0x27ece80(%rip), %rsi  # 0x27e334b <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x37f>
 4fd01cb:      	movl	$0x1, %edx
 4fd01d0:      	callq	*0x18(%rax)
 4fd01d3:      	andb	$0x1, %al
 4fd01d5:      	addq	$0x18, %rsp
 4fd01d9:      	popq	%rbx
 4fd01da:      	popq	%r12
 4fd01dc:      	popq	%r13
 4fd01de:      	popq	%r14
 4fd01e0:      	popq	%r15
 4fd01e2:      	popq	%rbp
 4fd01e3:      	retq
 4fd01e4:      	int3
 4fd01e5:      	int3
 4fd01e6:      	int3
 4fd01e7:      	int3
 4fd01e8:      	int3
 4fd01e9:      	int3
 4fd01ea:      	int3
 4fd01eb:      	int3
 4fd01ec:      	int3
 4fd01ed:      	int3
 4fd01ee:      	int3
 4fd01ef:      	int3

0000000004fd1340 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write>:
 4fd1340:      	pushq	%rbp
 4fd1341:      	movq	%rsp, %rbp
 4fd1344:      	pushq	%r15
 4fd1346:      	pushq	%r14
 4fd1348:      	pushq	%r13
 4fd134a:      	pushq	%r12
 4fd134c:      	pushq	%rbx
 4fd134d:      	subq	$0x38, %rsp
 4fd1351:      	movq	%rcx, %rbx
 4fd1354:      	movq	%rdx, %r13
 4fd1357:      	movq	%rdi, %r14
 4fd135a:      	testb	$0x1, %bl
 4fd135d:      	jne	0x4fd14cb <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x18b>
 4fd1363:      	movzbl	(%r13), %eax
 4fd1368:      	testb	%al, %al
 4fd136a:      	je	0x4fd14eb <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x1ab>
 4fd1370:      	movq	%rsi, -0x50(%rbp)
 4fd1374:      	movq	0x18(%rsi), %rcx
 4fd1378:      	movq	%rcx, -0x58(%rbp)
 4fd137c:      	movq	$0x0, -0x30(%rbp)
 4fd1384:      	jmp	0x4fd13a0 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x60>
 4fd1386:      	nopw	%cs:(%rax,%rax)
 4fd1390:      	movzbl	(%r12), %eax
 4fd1395:      	movq	%r12, %r13
 4fd1398:      	testb	%al, %al
 4fd139a:      	je	0x4fd14eb <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x1ab>
 4fd13a0:      	leaq	0x1(%r13), %r12
 4fd13a4:      	movzbl	%al, %r15d
 4fd13a8:      	testb	%al, %al
 4fd13aa:      	js	0x4fd13d0 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x90>
 4fd13ac:      	movq	%r14, %rdi
 4fd13af:      	movq	%r12, %rsi
 4fd13b2:      	movq	%r15, %rdx
 4fd13b5:      	callq	*-0x58(%rbp)
 4fd13b8:      	testb	%al, %al
 4fd13ba:      	jne	0x4fd14ef <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x1af>
 4fd13c0:      	addq	%r15, %r12
 4fd13c3:      	jmp	0x4fd1390 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x50>
 4fd13c5:      	nopw	%cs:(%rax,%rax)
 4fd13d0:      	cmpb	$-0x80, %al
 4fd13d2:      	je	0x4fd13f5 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0xb5>
 4fd13d4:      	cmpl	$0xc0, %r15d
 4fd13db:      	jne	0x4fd141c <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0xdc>
 4fd13dd:      	movq	-0x30(%rbp), %r15
 4fd13e1:      	movq	%r15, %rax
 4fd13e4:      	shlq	$0x4, %rax
 4fd13e8:      	movq	$0x60000020, -0x38(%rbp) # imm = 0x60000020
 4fd13f0:      	jmp	0x4fd1492 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x152>
 4fd13f5:      	movzwl	0x1(%r13), %r15d
 4fd13fa:      	leaq	0x3(%r13), %rsi
 4fd13fe:      	movq	%r14, %rdi
 4fd1401:      	movq	%r15, %rdx
 4fd1404:      	callq	*-0x58(%rbp)
 4fd1407:      	testb	%al, %al
 4fd1409:      	jne	0x4fd14ef <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x1af>
 4fd140f:      	leaq	(%r15,%r13), %r12
 4fd1413:      	addq	$0x3, %r12
 4fd1417:      	jmp	0x4fd1390 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x50>
 4fd141c:      	movl	$0x60000020, %ecx       # imm = 0x60000020
 4fd1421:      	testb	$0x1, %al
 4fd1423:      	je	0x4fd1430 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0xf0>
 4fd1425:      	movl	0x1(%r13), %ecx
 4fd1429:      	addq	$0x5, %r13
 4fd142d:      	movq	%r13, %r12
 4fd1430:      	testb	$0x2, %al
 4fd1432:      	jne	0x4fd144d <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x10d>
 4fd1434:      	xorl	%edx, %edx
 4fd1436:      	movq	-0x30(%rbp), %r15
 4fd143a:      	testb	$0x4, %al
 4fd143c:      	je	0x4fd145e <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x11e>
 4fd143e:      	movzwl	(%r12), %esi
 4fd1443:      	addq	$0x2, %r12
 4fd1447:      	testb	$0x8, %al
 4fd1449:      	jne	0x4fd1464 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x124>
 4fd144b:      	jmp	0x4fd146d <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x12d>
 4fd144d:      	movzwl	(%r12), %edx
 4fd1452:      	addq	$0x2, %r12
 4fd1456:      	movq	-0x30(%rbp), %r15
 4fd145a:      	testb	$0x4, %al
 4fd145c:      	jne	0x4fd143e <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0xfe>
 4fd145e:      	xorl	%esi, %esi
 4fd1460:      	testb	$0x8, %al
 4fd1462:      	je	0x4fd146d <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x12d>
 4fd1464:      	movzwl	(%r12), %r15d
 4fd1469:      	addq	$0x2, %r12
 4fd146d:      	testb	$0x10, %al
 4fd146f:      	jne	0x4fd14ba <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x17a>
 4fd1471:      	testb	$0x20, %al
 4fd1473:      	je	0x4fd1480 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x140>
 4fd1475:      	movzwl	%si, %eax
 4fd1478:      	shll	$0x4, %eax
 4fd147b:      	movzwl	0x8(%rbx,%rax), %esi
 4fd1480:      	movq	%r15, %rax
 4fd1483:      	shlq	$0x4, %rax
 4fd1487:      	movl	%ecx, -0x38(%rbp)
 4fd148a:      	movw	%dx, -0x34(%rbp)
 4fd148e:      	movw	%si, -0x32(%rbp)
 4fd1492:      	movq	%r14, -0x48(%rbp)
 4fd1496:      	movq	-0x50(%rbp), %rcx
 4fd149a:      	movq	%rcx, -0x40(%rbp)
 4fd149e:      	movq	(%rbx,%rax), %rdi
 4fd14a2:      	leaq	-0x48(%rbp), %rsi
 4fd14a6:      	callq	*0x8(%rbx,%rax)
 4fd14aa:      	testb	%al, %al
 4fd14ac:      	jne	0x4fd14ef <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x1af>
 4fd14ae:      	incq	%r15
 4fd14b1:      	movq	%r15, -0x30(%rbp)
 4fd14b5:      	jmp	0x4fd1390 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x50>
 4fd14ba:      	movzwl	%dx, %edx
 4fd14bd:      	shll	$0x4, %edx
 4fd14c0:      	movzwl	0x8(%rbx,%rdx), %edx
 4fd14c5:      	testb	$0x20, %al
 4fd14c7:      	jne	0x4fd1475 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x135>
 4fd14c9:      	jmp	0x4fd1480 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x140>
 4fd14cb:      	shrq	%rbx
 4fd14ce:      	movq	0x18(%rsi), %rax
 4fd14d2:      	movq	%r14, %rdi
 4fd14d5:      	movq	%r13, %rsi
 4fd14d8:      	movq	%rbx, %rdx
 4fd14db:      	addq	$0x38, %rsp
 4fd14df:      	popq	%rbx
 4fd14e0:      	popq	%r12
 4fd14e2:      	popq	%r13
 4fd14e4:      	popq	%r14
 4fd14e6:      	popq	%r15
 4fd14e8:      	popq	%rbp
 4fd14e9:      	jmpq	*%rax
 4fd14eb:      	xorl	%eax, %eax
 4fd14ed:      	jmp	0x4fd14f1 <_RNvNtCs4NRVxsYgnAr_4core3fmt5write+0x1b1>
 4fd14ef:      	movb	$0x1, %al
 4fd14f1:      	addq	$0x38, %rsp
 4fd14f5:      	popq	%rbx
 4fd14f6:      	popq	%r12
 4fd14f8:      	popq	%r13
 4fd14fa:      	popq	%r14
 4fd14fc:      	popq	%r15
 4fd14fe:      	popq	%rbp
 4fd14ff:      	retq

0000000004fd1970 <_RNvNtCs4NRVxsYgnAr_4core6option13expect_failed>:
 4fd1970:      	pushq	%rbp
 4fd1971:      	movq	%rsp, %rbp
 4fd1974:      	subq	$0x20, %rsp
 4fd1978:      	movq	%rdi, -0x10(%rbp)
 4fd197c:      	movq	%rsi, -0x8(%rbp)
 4fd1980:      	leaq	-0x10(%rbp), %rax
 4fd1984:      	movq	%rax, -0x20(%rbp)
 4fd1988:      	leaq	0x73a1(%rip), %rax      # 0x4fd8d30 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtB8_>
 4fd198f:      	movq	%rax, -0x18(%rbp)
 4fd1993:      	leaq	-0x4c20a69(%rip), %rdi  # 0x3b0f31 <anon.fa2e791037615afd04c732fea021b56d.1.llvm.7881675451276153>
 4fd199a:      	leaq	-0x20(%rbp), %rsi
 4fd199e:      	callq	*0x16f4dc(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd19a4:      	int3
 4fd19a5:      	int3
 4fd19a6:      	int3
 4fd19a7:      	int3
 4fd19a8:      	int3
 4fd19a9:      	int3
 4fd19aa:      	int3
 4fd19ab:      	int3
 4fd19ac:      	int3
 4fd19ad:      	int3
 4fd19ae:      	int3
 4fd19af:      	int3

0000000004fd19b0 <_RNvNtCs4NRVxsYgnAr_4core6option13unwrap_failed>:
 4fd19b0:      	pushq	%rbp
 4fd19b1:      	movq	%rsp, %rbp
 4fd19b4:      	movq	%rdi, %rdx
 4fd19b7:      	leaq	-0x27ee29c(%rip), %rdi  # 0x27e3722 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x756>
 4fd19be:      	movl	$0x2b, %esi
 4fd19c3:      	callq	*0x16f53f(%rip)         # 0x5140f08 <writev+0x5140f08>
 4fd19c9:      	int3
 4fd19ca:      	int3
 4fd19cb:      	int3
 4fd19cc:      	int3
 4fd19cd:      	int3
 4fd19ce:      	int3
 4fd19cf:      	int3

0000000004fd19d0 <_RNvNtCs4NRVxsYgnAr_4core6result13unwrap_failed>:
 4fd19d0:      	pushq	%rbp
 4fd19d1:      	movq	%rsp, %rbp
 4fd19d4:      	subq	$0x40, %rsp
 4fd19d8:      	movq	%rdi, -0x10(%rbp)
 4fd19dc:      	movq	%rsi, -0x8(%rbp)
 4fd19e0:      	movq	%rdx, -0x20(%rbp)
 4fd19e4:      	movq	%rcx, -0x18(%rbp)
 4fd19e8:      	leaq	-0x10(%rbp), %rax
 4fd19ec:      	movq	%rax, -0x40(%rbp)
 4fd19f0:      	leaq	0x7339(%rip), %rax      # 0x4fd8d30 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtB8_>
 4fd19f7:      	movq	%rax, -0x38(%rbp)
 4fd19fb:      	leaq	-0x20(%rbp), %rax
 4fd19ff:      	movq	%rax, -0x30(%rbp)
 4fd1a03:      	leaq	0x7096(%rip), %rax      # 0x4fd8aa0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtRDNtB6_5DebugEL_Bx_3fmtB8_>
 4fd1a0a:      	movq	%rax, -0x28(%rbp)
 4fd1a0e:      	leaq	-0x4c3fd45(%rip), %rdi  # 0x391cd0 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.251.llvm.5209651971967471703>
 4fd1a15:      	leaq	-0x40(%rbp), %rsi
 4fd1a19:      	movq	%r8, %rdx
 4fd1a1c:      	callq	*0x16f45e(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd1a22:      	int3
 4fd1a23:      	int3
 4fd1a24:      	int3
 4fd1a25:      	int3
 4fd1a26:      	int3
 4fd1a27:      	int3
 4fd1a28:      	int3
 4fd1a29:      	int3
 4fd1a2a:      	int3
 4fd1a2b:      	int3
 4fd1a2c:      	int3
 4fd1a2d:      	int3
 4fd1a2e:      	int3
 4fd1a2f:      	int3

0000000004fd1a30 <_RNvNtCs4NRVxsYgnAr_4core9panicking14panic_nounwind>:
 4fd1a30:      	pushq	%rbp
 4fd1a31:      	movq	%rsp, %rbp
 4fd1a34:      	leaq	0x1(,%rsi,2), %rsi
 4fd1a3c:      	leaq	0x16e8ad(%rip), %rcx    # 0x51402f0 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2cb8>
 4fd1a43:      	xorl	%edx, %edx
 4fd1a45:      	callq	*0x17aa25(%rip)         # 0x514c470 <writev+0x514c470>

0000000004fd1a4b <_RNvNtCs4NRVxsYgnAr_4core9panicking16panic_in_cleanup>:
 4fd1a4b:      	pushq	%rbp
 4fd1a4c:      	movq	%rsp, %rbp
 4fd1a4f:      	leaq	-0x27ee309(%rip), %rdi  # 0x27e374d <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x781>
 4fd1a56:      	pushq	$0x24
 4fd1a58:      	popq	%rsi
 4fd1a59:      	callq	*0x17aae9(%rip)         # 0x514c548 <writev+0x514c548>

0000000004fd1a5f <_RNvNtCs4NRVxsYgnAr_4core9panicking18panic_bounds_check>:
 4fd1a5f:      	pushq	%rbp
 4fd1a60:      	movq	%rsp, %rbp
 4fd1a63:      	subq	$0x30, %rsp
 4fd1a67:      	leaq	-0x8(%rbp), %rax
 4fd1a6b:      	movq	%rdi, (%rax)
 4fd1a6e:      	leaq	-0x10(%rbp), %rcx
 4fd1a72:      	movq	%rsi, (%rcx)
 4fd1a75:      	leaq	-0x30(%rbp), %rsi
 4fd1a79:      	movq	%rcx, (%rsi)
 4fd1a7c:      	movq	0x16f43d(%rip), %rcx    # 0x5140ec0 <writev+0x5140ec0>
 4fd1a83:      	movq	%rcx, 0x8(%rsi)
 4fd1a87:      	movq	%rax, 0x10(%rsi)
 4fd1a8b:      	movq	%rcx, 0x18(%rsi)
 4fd1a8f:      	leaq	-0x4c12e8a(%rip), %rdi  # 0x3bec0c <anon.8f057017deb32fa88495a56737728f46.6.llvm.16794444621540628717+0x180>
 4fd1a96:      	callq	*0x16f3e4(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd1a9c:      	int3
 4fd1a9d:      	int3
 4fd1a9e:      	int3
 4fd1a9f:      	int3

0000000004fd1aa0 <_RNvNtCs4NRVxsYgnAr_4core9panicking18panic_nounwind_fmt>:
 4fd1aa0:      	pushq	%rbp
 4fd1aa1:      	movq	%rsp, %rbp
 4fd1aa4:      	subq	$0x30, %rsp
 4fd1aa8:      	movq	%rdi, -0x28(%rbp)
 4fd1aac:      	movq	%rsi, -0x20(%rbp)
 4fd1ab0:      	leaq	-0x28(%rbp), %rax
 4fd1ab4:      	movq	%rax, -0x18(%rbp)
 4fd1ab8:      	movq	%rcx, -0x10(%rbp)
 4fd1abc:      	movb	$0x0, -0x8(%rbp)
 4fd1ac0:      	movb	%dl, -0x7(%rbp)
 4fd1ac3:      	leaq	-0x18(%rbp), %rdi
 4fd1ac7:      	callq	*0x16f233(%rip)         # 0x5140d00 <writev+0x5140d00>
 4fd1acd:      	ud2
 4fd1acf:      	callq	*0x16f34b(%rip)         # 0x5140e20 <writev+0x5140e20>

0000000004fd1ad5 <_RNvNtCs4NRVxsYgnAr_4core9panicking19assert_failed_inner>:
 4fd1ad5:      	pushq	%rbp
 4fd1ad6:      	movq	%rsp, %rbp
 4fd1ad9:      	subq	$0x80, %rsp
 4fd1ae0:      	movq	%rdx, %rax
 4fd1ae3:      	movq	0x18(%rbp), %rdx
 4fd1ae7:      	movq	%rsi, -0x10(%rbp)
 4fd1aeb:      	movq	%rax, -0x8(%rbp)
 4fd1aef:      	movq	%rcx, -0x20(%rbp)
 4fd1af3:      	movq	%r8, -0x18(%rbp)
 4fd1af7:      	leaq	-0x27e8d0a(%rip), %rax  # 0x27e8df4 <_RNvNtNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7flt2dec8strategy6dragon9POW5TO256+0x3a8>
 4fd1afe:      	movzbl	%dil, %ecx
 4fd1b02:      	movslq	(%rax,%rcx,4), %rsi
 4fd1b06:      	addq	%rax, %rsi
 4fd1b09:      	leaq	-0x27e8d10(%rip), %rax  # 0x27e8e00 <_RNvNtNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7flt2dec8strategy6dragon9POW5TO256+0x3b4>
 4fd1b10:      	movq	(%rax,%rcx,8), %rax
 4fd1b14:      	movq	%rsi, -0x30(%rbp)
 4fd1b18:      	movq	%rax, -0x28(%rbp)
 4fd1b1c:      	testq	%r9, %r9
 4fd1b1f:      	jne	0x4fd1b63 <_RNvNtCs4NRVxsYgnAr_4core9panicking19assert_failed_inner+0x8e>
 4fd1b21:      	leaq	-0x30(%rbp), %rax
 4fd1b25:      	leaq	-0x80(%rbp), %rsi
 4fd1b29:      	movq	%rax, (%rsi)
 4fd1b2c:      	leaq	0x71fd(%rip), %rax      # 0x4fd8d30 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtB8_>
 4fd1b33:      	movq	%rax, 0x8(%rsi)
 4fd1b37:      	leaq	-0x10(%rbp), %rax
 4fd1b3b:      	movq	%rax, 0x10(%rsi)
 4fd1b3f:      	leaq	0x6f5a(%rip), %rax      # 0x4fd8aa0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtRDNtB6_5DebugEL_Bx_3fmtB8_>
 4fd1b46:      	movq	%rax, 0x18(%rsi)
 4fd1b4a:      	leaq	-0x20(%rbp), %rcx
 4fd1b4e:      	movq	%rcx, 0x20(%rsi)
 4fd1b52:      	movq	%rax, 0x28(%rsi)
 4fd1b56:      	leaq	-0x4c34303(%rip), %rdi  # 0x39d85a <anon.e79e22280ec93d6ebd4bda234778991d.68.llvm.4565922734149631923+0x176>
 4fd1b5d:      	callq	*0x16f31d(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd1b63:      	movq	0x10(%rbp), %rax
 4fd1b67:      	leaq	-0x40(%rbp), %rcx
 4fd1b6b:      	movq	%r9, (%rcx)
 4fd1b6e:      	movq	%rax, 0x8(%rcx)
 4fd1b72:      	leaq	-0x30(%rbp), %rax
 4fd1b76:      	leaq	-0x80(%rbp), %rsi
 4fd1b7a:      	movq	%rax, (%rsi)
 4fd1b7d:      	leaq	0x71ac(%rip), %rax      # 0x4fd8d30 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtB8_>
 4fd1b84:      	movq	%rax, 0x8(%rsi)
 4fd1b88:      	movq	%rcx, 0x10(%rsi)
 4fd1b8c:      	movq	0x170fe5(%rip), %rax    # 0x5142b78 <writev+0x5142b78>
 4fd1b93:      	movq	%rax, 0x18(%rsi)
 4fd1b97:      	leaq	-0x10(%rbp), %rax
 4fd1b9b:      	movq	%rax, 0x20(%rsi)
 4fd1b9f:      	leaq	0x6efa(%rip), %rax      # 0x4fd8aa0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtRDNtB6_5DebugEL_Bx_3fmtB8_>
 4fd1ba6:      	movq	%rax, 0x28(%rsi)
 4fd1baa:      	leaq	-0x20(%rbp), %rcx
 4fd1bae:      	movq	%rcx, 0x30(%rsi)
 4fd1bb2:      	movq	%rax, 0x38(%rsi)
 4fd1bb6:      	leaq	-0x4c39bdc(%rip), %rdi  # 0x397fe1 <anon.3fed5a9ce85a26497d14c0bdddfc44ea.5.llvm.5251658545758491037+0x27d>
 4fd1bbd:      	callq	*0x16f2bd(%rip)         # 0x5140e80 <writev+0x5140e80>

0000000004fd1bc3 <_RNvNtCs4NRVxsYgnAr_4core9panicking19panic_cannot_unwind>:
 4fd1bc3:      	pushq	%rbp
 4fd1bc4:      	movq	%rsp, %rbp
 4fd1bc7:      	leaq	-0x27ee452(%rip), %rdi  # 0x27e377c <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x7b0>
 4fd1bce:      	pushq	$0x26
 4fd1bd0:      	popq	%rsi
 4fd1bd1:      	callq	*0x17a589(%rip)         # 0x514c160 <writev+0x514c160>
 4fd1bd7:      	int3
 4fd1bd8:      	int3
 4fd1bd9:      	int3
 4fd1bda:      	int3
 4fd1bdb:      	int3
 4fd1bdc:      	int3
 4fd1bdd:      	int3
 4fd1bde:      	int3
 4fd1bdf:      	int3

0000000004fd1be0 <_RNvNtCs4NRVxsYgnAr_4core9panicking26panic_nounwind_nobacktrace>:
 4fd1be0:      	pushq	%rbp
 4fd1be1:      	movq	%rsp, %rbp
 4fd1be4:      	leaq	0x1(,%rsi,2), %rsi
 4fd1bec:      	leaq	0x16e715(%rip), %rcx    # 0x5140308 <_RNvNtCskHv9yLCLqO6_13generic_array3hex11LOWER_CHARS+0x2cd0>
 4fd1bf3:      	movl	$0x1, %edx
 4fd1bf8:      	callq	*0x17a872(%rip)         # 0x514c470 <writev+0x514c470>
 4fd1bfe:      	int3
 4fd1bff:      	int3

0000000004fd1c00 <_RNvNtCs4NRVxsYgnAr_4core9panicking5panic>:
 4fd1c00:      	pushq	%rbp
 4fd1c01:      	movq	%rsp, %rbp
 4fd1c04:      	leaq	0x1(,%rsi,2), %rsi
 4fd1c0c:      	callq	*0x16f26e(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd1c12:      	int3
 4fd1c13:      	int3
 4fd1c14:      	int3
 4fd1c15:      	int3
 4fd1c16:      	int3
 4fd1c17:      	int3
 4fd1c18:      	int3
 4fd1c19:      	int3
 4fd1c1a:      	int3
 4fd1c1b:      	int3
 4fd1c1c:      	int3
 4fd1c1d:      	int3
 4fd1c1e:      	int3
 4fd1c1f:      	int3

0000000004fd1c20 <_RNvNtCs4NRVxsYgnAr_4core9panicking9panic_fmt>:
 4fd1c20:      	pushq	%rbp
 4fd1c21:      	movq	%rsp, %rbp
 4fd1c24:      	subq	$0x30, %rsp
 4fd1c28:      	movq	%rdi, -0x10(%rbp)
 4fd1c2c:      	movq	%rsi, -0x8(%rbp)
 4fd1c30:      	leaq	-0x10(%rbp), %rax
 4fd1c34:      	movq	%rax, -0x28(%rbp)
 4fd1c38:      	movq	%rdx, -0x20(%rbp)
 4fd1c3c:      	movw	$0x1, -0x18(%rbp)
 4fd1c42:      	leaq	-0x28(%rbp), %rdi
 4fd1c46:      	callq	*0x16f0b4(%rip)         # 0x5140d00 <writev+0x5140d00>
 4fd1c4c:      	int3
 4fd1c4d:      	int3
 4fd1c4e:      	int3
 4fd1c4f:      	int3

0000000004fd1c50 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars>:
 4fd1c50:      	leaq	0x7(%rdi), %r9
 4fd1c54:      	andq	$-0x8, %r9
 4fd1c58:      	movq	%r9, %r8
 4fd1c5b:      	subq	%rdi, %r8
 4fd1c5e:      	movq	%rsi, %rax
 4fd1c61:      	subq	%r8, %rax
 4fd1c64:      	jae	0x4fd1c7a <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x2a>
 4fd1c66:      	testq	%rsi, %rsi
 4fd1c69:      	je	0x4fd1ca8 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x58>
 4fd1c6b:      	cmpq	$0x4, %rsi
 4fd1c6f:      	jae	0x4fd1cab <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x5b>
 4fd1c71:      	xorl	%ecx, %ecx
 4fd1c73:      	xorl	%eax, %eax
 4fd1c75:      	jmp	0x4fd21f3 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x5a3>
 4fd1c7a:      	movq	%rax, %rcx
 4fd1c7d:      	shrq	$0x3, %rcx
 4fd1c81:      	je	0x4fd1c66 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x16>
 4fd1c83:      	movl	%eax, %edx
 4fd1c85:      	andl	$0x7, %edx
 4fd1c88:      	cmpq	%rdi, %r9
 4fd1c8b:      	jne	0x4fd1c94 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x44>
 4fd1c8d:      	xorl	%esi, %esi
 4fd1c8f:      	jmp	0x4fd1ddd <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x18d>
 4fd1c94:      	cmpq	$0x4, %r8
 4fd1c98:      	jae	0x4fd1d35 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0xe5>
 4fd1c9e:      	xorl	%r9d, %r9d
 4fd1ca1:      	xorl	%esi, %esi
 4fd1ca3:      	jmp	0x4fd1dc6 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x176>
 4fd1ca8:      	xorl	%eax, %eax
 4fd1caa:      	retq
 4fd1cab:      	movq	%rsi, %rcx
 4fd1cae:      	andq	$-0x4, %rcx
 4fd1cb2:      	pxor	%xmm0, %xmm0
 4fd1cb6:      	xorl	%eax, %eax
 4fd1cb8:      	movdqa	-0x4c4a530(%rip), %xmm1 # 0x387790 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x470>
 4fd1cc0:      	movdqa	-0x4c4a528(%rip), %xmm3 # 0x3877a0 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x480>
 4fd1cc8:      	pxor	%xmm2, %xmm2
 4fd1ccc:      	nopl	(%rax)
 4fd1cd0:      	movzwl	(%rdi,%rax), %edx
 4fd1cd4:      	movd	%edx, %xmm4
 4fd1cd8:      	movzwl	0x2(%rdi,%rax), %edx
 4fd1cdd:      	movd	%edx, %xmm5
 4fd1ce1:      	pcmpgtb	%xmm1, %xmm4
 4fd1ce5:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fd1ce9:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fd1cee:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fd1cf3:      	pand	%xmm3, %xmm4
 4fd1cf7:      	paddq	%xmm4, %xmm2
 4fd1cfb:      	pcmpgtb	%xmm1, %xmm5
 4fd1cff:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fd1d03:      	pshuflw	$0xd4, %xmm5, %xmm4     # xmm4 = xmm5[0,1,1,3,4,5,6,7]
 4fd1d08:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fd1d0d:      	pand	%xmm3, %xmm4
 4fd1d11:      	paddq	%xmm4, %xmm0
 4fd1d15:      	addq	$0x4, %rax
 4fd1d19:      	cmpq	%rax, %rcx
 4fd1d1c:      	jne	0x4fd1cd0 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x80>
 4fd1d1e:      	paddq	%xmm2, %xmm0
 4fd1d22:      	pshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
 4fd1d27:      	paddq	%xmm0, %xmm1
 4fd1d2b:      	movq	%xmm1, %rax
 4fd1d30:      	jmp	0x4fd2202 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x5b2>
 4fd1d35:      	movl	%r8d, %r9d
 4fd1d38:      	andl	$0x4, %r9d
 4fd1d3c:      	pxor	%xmm0, %xmm0
 4fd1d40:      	xorl	%esi, %esi
 4fd1d42:      	movdqa	-0x4c4a5ba(%rip), %xmm1 # 0x387790 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x470>
 4fd1d4a:      	movdqa	-0x4c4a5b2(%rip), %xmm3 # 0x3877a0 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x480>
 4fd1d52:      	pxor	%xmm2, %xmm2
 4fd1d56:      	nopw	%cs:(%rax,%rax)
 4fd1d60:      	movzwl	(%rdi,%rsi), %r10d
 4fd1d65:      	movd	%r10d, %xmm4
 4fd1d6a:      	movzwl	0x2(%rdi,%rsi), %r10d
 4fd1d70:      	movd	%r10d, %xmm5
 4fd1d75:      	pcmpgtb	%xmm1, %xmm4
 4fd1d79:      	punpcklbw	%xmm4, %xmm4    # xmm4 = xmm4[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fd1d7d:      	pshuflw	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3,4,5,6,7]
 4fd1d82:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fd1d87:      	pand	%xmm3, %xmm4
 4fd1d8b:      	paddq	%xmm4, %xmm2
 4fd1d8f:      	pcmpgtb	%xmm1, %xmm5
 4fd1d93:      	punpcklbw	%xmm5, %xmm5    # xmm5 = xmm5[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fd1d97:      	pshuflw	$0xd4, %xmm5, %xmm4     # xmm4 = xmm5[0,1,1,3,4,5,6,7]
 4fd1d9c:      	pshufd	$0xd4, %xmm4, %xmm4     # xmm4 = xmm4[0,1,1,3]
 4fd1da1:      	pand	%xmm3, %xmm4
 4fd1da5:      	paddq	%xmm4, %xmm0
 4fd1da9:      	addq	$0x4, %rsi
 4fd1dad:      	cmpq	%rsi, %r9
 4fd1db0:      	jne	0x4fd1d60 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x110>
 4fd1db2:      	paddq	%xmm2, %xmm0
 4fd1db6:      	pshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
 4fd1dbb:      	paddq	%xmm0, %xmm1
 4fd1dbf:      	movq	%xmm1, %rsi
 4fd1dc4:      	jmp	0x4fd1dd8 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x188>
 4fd1dc6:      	xorl	%r10d, %r10d
 4fd1dc9:      	cmpb	$-0x40, (%rdi,%r9)
 4fd1dce:      	setge	%r10b
 4fd1dd2:      	addq	%r10, %rsi
 4fd1dd5:      	incq	%r9
 4fd1dd8:      	cmpq	%r9, %r8
 4fd1ddb:      	jne	0x4fd1dc6 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x176>
 4fd1ddd:      	addq	%r8, %rdi
 4fd1de0:      	testq	%rdx, %rdx
 4fd1de3:      	je	0x4fd1e79 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x229>
 4fd1de9:      	movabsq	$0x7ffffffffffffff8, %r8 # imm = 0x7FFFFFFFFFFFFFF8
 4fd1df3:      	andq	%r8, %rax
 4fd1df6:      	xorl	%r10d, %r10d
 4fd1df9:      	cmpb	$-0x40, (%rdi,%rax)
 4fd1dfd:      	setge	%r10b
 4fd1e01:      	cmpl	$0x1, %edx
 4fd1e04:      	je	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e06:      	xorl	%r8d, %r8d
 4fd1e09:      	cmpb	$-0x40, 0x1(%rdi,%rax)
 4fd1e0e:      	setge	%r8b
 4fd1e12:      	addq	%r8, %r10
 4fd1e15:      	cmpl	$0x2, %edx
 4fd1e18:      	je	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e1a:      	xorl	%r8d, %r8d
 4fd1e1d:      	cmpb	$-0x40, 0x2(%rdi,%rax)
 4fd1e22:      	setge	%r8b
 4fd1e26:      	addq	%r8, %r10
 4fd1e29:      	cmpl	$0x3, %edx
 4fd1e2c:      	je	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e2e:      	xorl	%r8d, %r8d
 4fd1e31:      	cmpb	$-0x40, 0x3(%rdi,%rax)
 4fd1e36:      	setge	%r8b
 4fd1e3a:      	addq	%r8, %r10
 4fd1e3d:      	cmpl	$0x4, %edx
 4fd1e40:      	je	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e42:      	xorl	%r8d, %r8d
 4fd1e45:      	cmpb	$-0x40, 0x4(%rdi,%rax)
 4fd1e4a:      	setge	%r8b
 4fd1e4e:      	addq	%r8, %r10
 4fd1e51:      	cmpl	$0x5, %edx
 4fd1e54:      	je	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e56:      	xorl	%r8d, %r8d
 4fd1e59:      	cmpb	$-0x40, 0x5(%rdi,%rax)
 4fd1e5e:      	setge	%r8b
 4fd1e62:      	addq	%r8, %r10
 4fd1e65:      	cmpl	$0x6, %edx
 4fd1e68:      	je	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e6a:      	xorl	%edx, %edx
 4fd1e6c:      	cmpb	$-0x40, 0x6(%rdi,%rax)
 4fd1e71:      	setge	%dl
 4fd1e74:      	addq	%rdx, %r10
 4fd1e77:      	jmp	0x4fd1e7c <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x22c>
 4fd1e79:      	xorl	%r10d, %r10d
 4fd1e7c:      	pushq	%r15
 4fd1e7e:      	pushq	%r14
 4fd1e80:      	pushq	%r12
 4fd1e82:      	pushq	%rbx
 4fd1e83:      	addq	%rsi, %r10
 4fd1e86:      	movabsq	$0xff00ff00ff00ff, %rsi # imm = 0xFF00FF00FF00FF
 4fd1e90:      	movabsq	$0x1000100010001, %rdx  # imm = 0x1000100010001
 4fd1e9a:      	pcmpeqd	%xmm0, %xmm0
 4fd1e9e:      	movdqa	-0x4c513f6(%rip), %xmm1 # 0x380ab0 <anon.d42d8d1710be02f83480d8869995a290.139.llvm.15708504647765068451+0x1a0>
 4fd1ea6:      	jmp	0x4fd1ee4 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x294>
 4fd1ea8:      	nopl	(%rax,%rax)
 4fd1eb0:      	xorl	%r10d, %r10d
 4fd1eb3:      	addq	%r8, %rdi
 4fd1eb6:      	subq	%r9, %rcx
 4fd1eb9:      	movl	%r9d, %r11d
 4fd1ebc:      	andl	$0x3, %r11d
 4fd1ec0:      	movq	%r10, %rbx
 4fd1ec3:      	andq	%rsi, %rbx
 4fd1ec6:      	shrq	$0x8, %r10
 4fd1eca:      	andq	%rsi, %r10
 4fd1ecd:      	addq	%rbx, %r10
 4fd1ed0:      	imulq	%rdx, %r10
 4fd1ed4:      	shrq	$0x30, %r10
 4fd1ed8:      	addq	%rax, %r10
 4fd1edb:      	testq	%r11, %r11
 4fd1ede:      	jne	0x4fd2160 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x510>
 4fd1ee4:      	movq	%r10, %rax
 4fd1ee7:      	testq	%rcx, %rcx
 4fd1eea:      	je	0x4fd21eb <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x59b>
 4fd1ef0:      	movq	%rdi, %r8
 4fd1ef3:      	cmpq	$0xc0, %rcx
 4fd1efa:      	movl	$0xc0, %r9d
 4fd1f00:      	cmovbq	%rcx, %r9
 4fd1f04:      	leal	(,%r9,8), %edi
 4fd1f0c:      	movl	%edi, %r11d
 4fd1f0f:      	andl	$0x7e0, %r11d           # imm = 0x7E0
 4fd1f16:      	je	0x4fd1eb0 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x260>
 4fd1f18:      	leaq	-0x20(%rdi), %r14
 4fd1f1c:      	cmpq	$0x60, %r14
 4fd1f20:      	jae	0x4fd1f30 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x2e0>
 4fd1f22:      	movq	%r8, %rbx
 4fd1f25:      	xorl	%r10d, %r10d
 4fd1f28:      	jmp	0x4fd20f8 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x4a8>
 4fd1f2d:      	nopl	(%rax)
 4fd1f30:      	shrq	$0x5, %r14
 4fd1f34:      	incq	%r14
 4fd1f37:      	movq	%r14, %r15
 4fd1f3a:      	andq	$-0x4, %r15
 4fd1f3e:      	movq	%r15, %rbx
 4fd1f41:      	shlq	$0x5, %rbx
 4fd1f45:      	addq	%r8, %rbx
 4fd1f48:      	leaq	0x40(%r8), %r10
 4fd1f4c:      	pxor	%xmm4, %xmm4
 4fd1f50:      	movq	%r15, %r12
 4fd1f53:      	pxor	%xmm6, %xmm6
 4fd1f57:      	nopw	(%rax,%rax)
 4fd1f60:      	movdqu	-0x40(%r10), %xmm2
 4fd1f66:      	movdqu	-0x20(%r10), %xmm5
 4fd1f6c:      	movdqa	%xmm2, %xmm3
 4fd1f70:      	punpcklqdq	%xmm5, %xmm3    # xmm3 = xmm3[0],xmm5[0]
 4fd1f74:      	movdqa	%xmm3, %xmm7
 4fd1f78:      	pxor	%xmm0, %xmm7
 4fd1f7c:      	psrlq	$0x7, %xmm7
 4fd1f81:      	psrlq	$0x6, %xmm3
 4fd1f86:      	por	%xmm7, %xmm3
 4fd1f8a:      	pand	%xmm1, %xmm3
 4fd1f8e:      	paddq	%xmm4, %xmm3
 4fd1f92:      	movdqu	-0x30(%r10), %xmm4
 4fd1f98:      	movdqu	-0x10(%r10), %xmm7
 4fd1f9e:      	punpckhqdq	%xmm5, %xmm2    # xmm2 = xmm2[1],xmm5[1]
 4fd1fa2:      	movdqa	%xmm4, %xmm5
 4fd1fa6:      	punpcklqdq	%xmm7, %xmm5    # xmm5 = xmm5[0],xmm7[0]
 4fd1faa:      	punpckhqdq	%xmm7, %xmm4    # xmm4 = xmm4[1],xmm7[1]
 4fd1fae:      	movdqu	0x30(%r10), %xmm10
 4fd1fb4:      	movdqu	0x20(%r10), %xmm7
 4fd1fba:      	movdqu	(%r10), %xmm9
 4fd1fbf:      	movdqa	%xmm9, %xmm8
 4fd1fc4:      	punpcklqdq	%xmm7, %xmm8    # xmm8 = xmm8[0],xmm7[0]
 4fd1fc9:      	movdqa	%xmm8, %xmm11
 4fd1fce:      	pxor	%xmm0, %xmm11
 4fd1fd3:      	psrlq	$0x7, %xmm11
 4fd1fd9:      	psrlq	$0x6, %xmm8
 4fd1fdf:      	por	%xmm11, %xmm8
 4fd1fe4:      	pand	%xmm1, %xmm8
 4fd1fe9:      	paddq	%xmm6, %xmm8
 4fd1fee:      	movdqu	0x10(%r10), %xmm6
 4fd1ff4:      	punpckhqdq	%xmm7, %xmm9    # xmm9 = xmm9[1],xmm7[1]
 4fd1ff9:      	movdqa	%xmm6, %xmm7
 4fd1ffd:      	punpcklqdq	%xmm10, %xmm7   # xmm7 = xmm7[0],xmm10[0]
 4fd2002:      	punpckhqdq	%xmm10, %xmm6   # xmm6 = xmm6[1],xmm10[1]
 4fd2007:      	movdqa	%xmm2, %xmm10
 4fd200c:      	pxor	%xmm0, %xmm10
 4fd2011:      	movdqa	%xmm9, %xmm11
 4fd2016:      	pxor	%xmm0, %xmm11
 4fd201b:      	psrlq	$0x7, %xmm10
 4fd2021:      	psrlq	$0x7, %xmm11
 4fd2027:      	psrlq	$0x6, %xmm2
 4fd202c:      	por	%xmm10, %xmm2
 4fd2031:      	psrlq	$0x6, %xmm9
 4fd2037:      	por	%xmm11, %xmm9
 4fd203c:      	pand	%xmm1, %xmm2
 4fd2040:      	pand	%xmm1, %xmm9
 4fd2045:      	movdqa	%xmm5, %xmm10
 4fd204a:      	pxor	%xmm0, %xmm10
 4fd204f:      	movdqa	%xmm7, %xmm11
 4fd2054:      	pxor	%xmm0, %xmm11
 4fd2059:      	psrlq	$0x7, %xmm10
 4fd205f:      	psrlq	$0x7, %xmm11
 4fd2065:      	psrlq	$0x6, %xmm5
 4fd206a:      	por	%xmm10, %xmm5
 4fd206f:      	psrlq	$0x6, %xmm7
 4fd2074:      	por	%xmm11, %xmm7
 4fd2079:      	pand	%xmm1, %xmm5
 4fd207d:      	paddq	%xmm2, %xmm5
 4fd2081:      	paddq	%xmm3, %xmm5
 4fd2085:      	pand	%xmm1, %xmm7
 4fd2089:      	paddq	%xmm9, %xmm7
 4fd208e:      	paddq	%xmm8, %xmm7
 4fd2093:      	movdqa	%xmm4, %xmm2
 4fd2097:      	pxor	%xmm0, %xmm2
 4fd209b:      	movdqa	%xmm6, %xmm3
 4fd209f:      	pxor	%xmm0, %xmm3
 4fd20a3:      	psrlq	$0x7, %xmm2
 4fd20a8:      	psrlq	$0x7, %xmm3
 4fd20ad:      	psrlq	$0x6, %xmm4
 4fd20b2:      	por	%xmm2, %xmm4
 4fd20b6:      	psrlq	$0x6, %xmm6
 4fd20bb:      	por	%xmm3, %xmm6
 4fd20bf:      	pand	%xmm1, %xmm4
 4fd20c3:      	paddq	%xmm5, %xmm4
 4fd20c7:      	pand	%xmm1, %xmm6
 4fd20cb:      	paddq	%xmm7, %xmm6
 4fd20cf:      	subq	$-0x80, %r10
 4fd20d3:      	addq	$-0x4, %r12
 4fd20d7:      	jne	0x4fd1f60 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x310>
 4fd20dd:      	paddq	%xmm4, %xmm6
 4fd20e1:      	pshufd	$0xee, %xmm6, %xmm2     # xmm2 = xmm6[2,3,2,3]
 4fd20e6:      	paddq	%xmm6, %xmm2
 4fd20ea:      	movq	%xmm2, %r10
 4fd20ef:      	cmpq	%r15, %r14
 4fd20f2:      	je	0x4fd1eb3 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x263>
 4fd20f8:      	addq	%r8, %r11
 4fd20fb:      	nopl	(%rax,%rax)
 4fd2100:      	movdqu	(%rbx), %xmm2
 4fd2104:      	movdqu	0x10(%rbx), %xmm3
 4fd2109:      	movdqa	%xmm2, %xmm4
 4fd210d:      	pxor	%xmm0, %xmm4
 4fd2111:      	movdqa	%xmm3, %xmm5
 4fd2115:      	pxor	%xmm0, %xmm5
 4fd2119:      	psrlq	$0x7, %xmm5
 4fd211e:      	psrlq	$0x7, %xmm4
 4fd2123:      	psrlq	$0x6, %xmm3
 4fd2128:      	por	%xmm5, %xmm3
 4fd212c:      	psrlq	$0x6, %xmm2
 4fd2131:      	por	%xmm4, %xmm2
 4fd2135:      	pand	%xmm1, %xmm3
 4fd2139:      	pand	%xmm1, %xmm2
 4fd213d:      	paddq	%xmm3, %xmm2
 4fd2141:      	pshufd	$0xee, %xmm2, %xmm3     # xmm3 = xmm2[2,3,2,3]
 4fd2146:      	paddq	%xmm2, %xmm3
 4fd214a:      	movq	%xmm3, %r14
 4fd214f:      	addq	%r14, %r10
 4fd2152:      	addq	$0x20, %rbx
 4fd2156:      	cmpq	%r11, %rbx
 4fd2159:      	jne	0x4fd2100 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x4b0>
 4fd215b:      	jmp	0x4fd1eb3 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x263>
 4fd2160:      	movabsq	$0x101010101010101, %rcx # imm = 0x101010101010101
 4fd216a:      	andl	$0xfc, %r9d
 4fd2171:      	movl	%r9d, %eax
 4fd2174:      	movq	(%r8,%rax,8), %rax
 4fd2178:      	movq	%rax, %rdi
 4fd217b:      	notq	%rdi
 4fd217e:      	shrq	$0x7, %rdi
 4fd2182:      	shrq	$0x6, %rax
 4fd2186:      	orq	%rdi, %rax
 4fd2189:      	andq	%rcx, %rax
 4fd218c:      	cmpl	$0x1, %r11d
 4fd2190:      	je	0x4fd21d0 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x580>
 4fd2192:      	movq	0x8(%r8,%r9,8), %rdi
 4fd2197:      	movq	%rdi, %rbx
 4fd219a:      	notq	%rbx
 4fd219d:      	shrq	$0x7, %rbx
 4fd21a1:      	shrq	$0x6, %rdi
 4fd21a5:      	orq	%rbx, %rdi
 4fd21a8:      	andq	%rcx, %rdi
 4fd21ab:      	addq	%rdi, %rax
 4fd21ae:      	cmpl	$0x2, %r11d
 4fd21b2:      	je	0x4fd21d0 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x580>
 4fd21b4:      	movq	0x10(%r8,%r9,8), %rdi
 4fd21b9:      	movq	%rdi, %r8
 4fd21bc:      	notq	%r8
 4fd21bf:      	shrq	$0x7, %r8
 4fd21c3:      	shrq	$0x6, %rdi
 4fd21c7:      	orq	%r8, %rdi
 4fd21ca:      	andq	%rcx, %rdi
 4fd21cd:      	addq	%rdi, %rax
 4fd21d0:      	movq	%rax, %rcx
 4fd21d3:      	andq	%rsi, %rcx
 4fd21d6:      	shrq	$0x8, %rax
 4fd21da:      	andq	%rsi, %rax
 4fd21dd:      	addq	%rcx, %rax
 4fd21e0:      	imulq	%rdx, %rax
 4fd21e4:      	shrq	$0x30, %rax
 4fd21e8:      	addq	%r10, %rax
 4fd21eb:      	popq	%rbx
 4fd21ec:      	popq	%r12
 4fd21ee:      	popq	%r14
 4fd21f0:      	popq	%r15
 4fd21f2:      	retq
 4fd21f3:      	xorl	%edx, %edx
 4fd21f5:      	cmpb	$-0x40, (%rdi,%rcx)
 4fd21f9:      	setge	%dl
 4fd21fc:      	addq	%rdx, %rax
 4fd21ff:      	incq	%rcx
 4fd2202:      	cmpq	%rcx, %rsi
 4fd2205:      	jne	0x4fd21f3 <_RNvNtNtCs4NRVxsYgnAr_4core3str5count14do_count_chars+0x5a3>
 4fd2207:      	retq
 4fd2208:      	int3
 4fd2209:      	int3
 4fd220a:      	int3
 4fd220b:      	int3
 4fd220c:      	int3
 4fd220d:      	int3
 4fd220e:      	int3
 4fd220f:      	int3

0000000004fd22d0 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8>:
 4fd22d0:      	pushq	%r15
 4fd22d2:      	pushq	%r14
 4fd22d4:      	pushq	%rbx
 4fd22d5:      	movq	%rdi, %rax
 4fd22d8:      	xorl	%edi, %edi
 4fd22da:      	movq	%rdx, %rcx
 4fd22dd:      	subq	$0xf, %rcx
 4fd22e1:      	cmovaeq	%rcx, %rdi
 4fd22e5:      	testq	%rdx, %rdx
 4fd22e8:      	je	0x4fd2497 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1c7>
 4fd22ee:      	leaq	0x7(%rsi), %r8
 4fd22f2:      	andq	$-0x8, %r8
 4fd22f6:      	subq	%rsi, %r8
 4fd22f9:      	xorl	%ecx, %ecx
 4fd22fb:      	leaq	-0x27eea95(%rip), %r10  # 0x27e386d <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x8a1>
 4fd2302:      	movabsq	$-0x7f7f7f7f7f7f7f80, %r11 # imm = 0x8080808080808080
 4fd230c:      	jmp	0x4fd231f <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x4f>
 4fd230e:      	nop
 4fd2310:      	incq	%r14
 4fd2313:      	movq	%r14, %rcx
 4fd2316:      	cmpq	%rdx, %rcx
 4fd2319:      	jae	0x4fd2497 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1c7>
 4fd231f:      	movzbl	(%rsi,%rcx), %ebx
 4fd2323:      	testb	%bl, %bl
 4fd2325:      	js	0x4fd2380 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0xb0>
 4fd2327:      	movl	%r8d, %r9d
 4fd232a:      	subl	%ecx, %r9d
 4fd232d:      	testb	$0x7, %r9b
 4fd2331:      	je	0x4fd2344 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x74>
 4fd2333:      	incq	%rcx
 4fd2336:      	jmp	0x4fd2316 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x46>
 4fd2338:      	nopl	(%rax,%rax)
 4fd2340:      	addq	$0x10, %rcx
 4fd2344:      	cmpq	%rdi, %rcx
 4fd2347:      	jae	0x4fd2357 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x87>
 4fd2349:      	movq	0x8(%rsi,%rcx), %r9
 4fd234e:      	orq	(%rsi,%rcx), %r9
 4fd2352:      	testq	%r11, %r9
 4fd2355:      	je	0x4fd2340 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x70>
 4fd2357:      	cmpq	%rdx, %rcx
 4fd235a:      	jae	0x4fd2316 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x46>
 4fd235c:      	nopl	(%rax)
 4fd2360:      	cmpb	$0x0, (%rsi,%rcx)
 4fd2364:      	js	0x4fd2316 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x46>
 4fd2366:      	incq	%rcx
 4fd2369:      	cmpq	%rcx, %rdx
 4fd236c:      	jne	0x4fd2360 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x90>
 4fd236e:      	jmp	0x4fd2497 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1c7>
 4fd2373:      	nopw	%cs:(%rax,%rax)
 4fd2380:      	movzbl	(%rbx,%r10), %r14d
 4fd2385:      	movb	$0x1, %r9b
 4fd2388:      	cmpl	$0x4, %r14d
 4fd238c:      	je	0x4fd23eb <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x11b>
 4fd238e:      	cmpl	$0x3, %r14d
 4fd2392:      	je	0x4fd23bd <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0xed>
 4fd2394:      	cmpl	$0x2, %r14d
 4fd2398:      	jne	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd239e:      	leaq	0x1(%rcx), %r14
 4fd23a2:      	cmpq	%rdx, %r14
 4fd23a5:      	jae	0x4fd24a3 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d3>
 4fd23ab:      	cmpb	$-0x41, (%rsi,%r14)
 4fd23b0:      	movb	$0x1, %bl
 4fd23b2:      	jle	0x4fd2310 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x40>
 4fd23b8:      	jmp	0x4fd24ae <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1de>
 4fd23bd:      	leaq	0x1(%rcx), %r14
 4fd23c1:      	cmpq	%rdx, %r14
 4fd23c4:      	jae	0x4fd24a3 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d3>
 4fd23ca:      	movzbl	(%rsi,%r14), %r14d
 4fd23cf:      	cmpq	$0xe0, %rbx
 4fd23d6:      	je	0x4fd2419 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x149>
 4fd23d8:      	cmpl	$0xed, %ebx
 4fd23de:      	jne	0x4fd2434 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x164>
 4fd23e0:      	cmpb	$-0x61, %r14b
 4fd23e4:      	jle	0x4fd244c <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x17c>
 4fd23e6:      	jmp	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd23eb:      	leaq	0x1(%rcx), %r14
 4fd23ef:      	cmpq	%rdx, %r14
 4fd23f2:      	jae	0x4fd24a3 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d3>
 4fd23f8:      	movzbl	(%rsi,%r14), %r14d
 4fd23fd:      	cmpq	$0xf0, %rbx
 4fd2404:      	je	0x4fd2428 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x158>
 4fd2406:      	cmpl	$0xf4, %ebx
 4fd240c:      	jne	0x4fd2462 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x192>
 4fd240e:      	cmpb	$-0x71, %r14b
 4fd2412:      	jle	0x4fd2470 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1a0>
 4fd2414:      	jmp	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd2419:      	andb	$-0x20, %r14b
 4fd241d:      	cmpb	$-0x60, %r14b
 4fd2421:      	je	0x4fd244c <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x17c>
 4fd2423:      	jmp	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd2428:      	addb	$0x70, %r14b
 4fd242c:      	cmpb	$0x30, %r14b
 4fd2430:      	jb	0x4fd2470 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1a0>
 4fd2432:      	jmp	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd2434:      	leal	0x1f(%rbx), %r15d
 4fd2438:      	cmpb	$0xc, %r15b
 4fd243c:      	jb	0x4fd2446 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x176>
 4fd243e:      	andb	$-0x2, %bl
 4fd2441:      	cmpb	$-0x12, %bl
 4fd2444:      	jne	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd2446:      	cmpb	$-0x40, %r14b
 4fd244a:      	jge	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd244c:      	leaq	0x2(%rcx), %r14
 4fd2450:      	cmpq	%rdx, %r14
 4fd2453:      	jae	0x4fd24a3 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d3>
 4fd2455:      	cmpb	$-0x41, (%rsi,%r14)
 4fd245a:      	jle	0x4fd2310 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x40>
 4fd2460:      	jmp	0x4fd24ac <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1dc>
 4fd2462:      	addb	$0xf, %bl
 4fd2465:      	cmpb	$0x2, %bl
 4fd2468:      	ja	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd246a:      	cmpb	$-0x40, %r14b
 4fd246e:      	jge	0x4fd24a8 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d8>
 4fd2470:      	leaq	0x2(%rcx), %rbx
 4fd2474:      	cmpq	%rdx, %rbx
 4fd2477:      	jae	0x4fd24a3 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d3>
 4fd2479:      	cmpb	$-0x41, (%rsi,%rbx)
 4fd247d:      	jg	0x4fd24ac <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1dc>
 4fd247f:      	leaq	0x3(%rcx), %r14
 4fd2483:      	cmpq	%rdx, %r14
 4fd2486:      	jae	0x4fd24a3 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1d3>
 4fd2488:      	cmpb	$-0x40, (%rsi,%r14)
 4fd248d:      	jl	0x4fd2310 <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x40>
 4fd2493:      	movb	$0x3, %bl
 4fd2495:      	jmp	0x4fd24ae <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1de>
 4fd2497:      	movq	%rsi, 0x8(%rax)
 4fd249b:      	movq	%rdx, 0x10(%rax)
 4fd249f:      	xorl	%ecx, %ecx
 4fd24a1:      	jmp	0x4fd24be <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1ee>
 4fd24a3:      	xorl	%r9d, %r9d
 4fd24a6:      	jmp	0x4fd24ae <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1de>
 4fd24a8:      	movb	$0x1, %bl
 4fd24aa:      	jmp	0x4fd24ae <_RNvNtNtCs4NRVxsYgnAr_4core3str8converts9from_utf8+0x1de>
 4fd24ac:      	movb	$0x2, %bl
 4fd24ae:      	movq	%rcx, 0x8(%rax)
 4fd24b2:      	movb	%r9b, 0x10(%rax)
 4fd24b6:      	movb	%bl, 0x11(%rax)
 4fd24b9:      	movl	$0x1, %ecx
 4fd24be:      	movq	%rcx, (%rax)
 4fd24c1:      	popq	%rbx
 4fd24c2:      	popq	%r14
 4fd24c4:      	popq	%r15
 4fd24c6:      	retq
 4fd24c7:      	int3
 4fd24c8:      	int3
 4fd24c9:      	int3
 4fd24ca:      	int3
 4fd24cb:      	int3
 4fd24cc:      	int3
 4fd24cd:      	int3
 4fd24ce:      	int3
 4fd24cf:      	int3

0000000004fd24d0 <_RNvNtNtCs4NRVxsYgnAr_4core5slice5index16slice_index_fail>:
 4fd24d0:      	pushq	%rbp
 4fd24d1:      	movq	%rsp, %rbp
 4fd24d4:      	subq	$0x30, %rsp
 4fd24d8:      	cmpq	%rdx, %rdi
 4fd24db:      	jbe	0x4fd2518 <_RNvNtNtCs4NRVxsYgnAr_4core5slice5index16slice_index_fail+0x48>
 4fd24dd:      	movq	%rdi, -0x8(%rbp)
 4fd24e1:      	movq	%rdx, -0x10(%rbp)
 4fd24e5:      	leaq	-0x8(%rbp), %rax
 4fd24e9:      	movq	%rax, -0x30(%rbp)
 4fd24ed:      	movq	0x16e9cc(%rip), %rax    # 0x5140ec0 <writev+0x5140ec0>
 4fd24f4:      	movq	%rax, -0x28(%rbp)
 4fd24f8:      	leaq	-0x10(%rbp), %rdx
 4fd24fc:      	movq	%rdx, -0x20(%rbp)
 4fd2500:      	movq	%rax, -0x18(%rbp)
 4fd2504:      	leaq	-0x4bfc9dd(%rip), %rdi  # 0x3d5b2e <anon.e79e22280ec93d6ebd4bda234778991d.24.llvm.4565922734149631923+0x1b9>
 4fd250b:      	leaq	-0x30(%rbp), %rsi
 4fd250f:      	movq	%rcx, %rdx
 4fd2512:      	callq	*0x16e968(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd2518:      	cmpq	%rdx, %rsi
 4fd251b:      	ja	0x4fd255d <_RNvNtNtCs4NRVxsYgnAr_4core5slice5index16slice_index_fail+0x8d>
 4fd251d:      	cmpq	%rsi, %rdi
 4fd2520:      	jbe	0x4fd255d <_RNvNtNtCs4NRVxsYgnAr_4core5slice5index16slice_index_fail+0x8d>
 4fd2522:      	movq	%rdi, -0x8(%rbp)
 4fd2526:      	movq	%rsi, -0x10(%rbp)
 4fd252a:      	leaq	-0x8(%rbp), %rax
 4fd252e:      	movq	%rax, -0x30(%rbp)
 4fd2532:      	movq	0x16e987(%rip), %rax    # 0x5140ec0 <writev+0x5140ec0>
 4fd2539:      	movq	%rax, -0x28(%rbp)
 4fd253d:      	leaq	-0x10(%rbp), %rdx
 4fd2541:      	movq	%rdx, -0x20(%rbp)
 4fd2545:      	movq	%rax, -0x18(%rbp)
 4fd2549:      	leaq	-0x4c19411(%rip), %rdi  # 0x3b913f <anon.ea9ff54ec891c523b22d7beb433c6217.1.llvm.10671892735646583555+0xeb>
 4fd2550:      	leaq	-0x30(%rbp), %rsi
 4fd2554:      	movq	%rcx, %rdx
 4fd2557:      	callq	*0x16e923(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd255d:      	movq	%rsi, -0x8(%rbp)
 4fd2561:      	movq	%rdx, -0x10(%rbp)
 4fd2565:      	leaq	-0x8(%rbp), %rax
 4fd2569:      	movq	%rax, -0x30(%rbp)
 4fd256d:      	movq	0x16e94c(%rip), %rax    # 0x5140ec0 <writev+0x5140ec0>
 4fd2574:      	movq	%rax, -0x28(%rbp)
 4fd2578:      	leaq	-0x10(%rbp), %rdx
 4fd257c:      	movq	%rdx, -0x20(%rbp)
 4fd2580:      	movq	%rax, -0x18(%rbp)
 4fd2584:      	leaq	-0x4c020ab(%rip), %rdi  # 0x3d04e0 <anon.f4568be7443aebc71d53df3c2cdc86c9.15.llvm.6149804887664172415+0xf7>
 4fd258b:      	leaq	-0x30(%rbp), %rsi
 4fd258f:      	movq	%rcx, %rdx
 4fd2592:      	callq	*0x16e8e8(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd2598:      	int3
 4fd2599:      	int3
 4fd259a:      	int3
 4fd259b:      	int3
 4fd259c:      	int3
 4fd259d:      	int3
 4fd259e:      	int3
 4fd259f:      	int3

0000000004fd2a50 <_RNvNtNtCs4NRVxsYgnAr_4core9panicking11panic_const23panic_const_div_by_zero>:
 4fd2a50:      	pushq	%rbp
 4fd2a51:      	movq	%rsp, %rbp
 4fd2a54:      	movq	%rdi, %rdx
 4fd2a57:      	leaq	-0x27eeb3d(%rip), %rdi  # 0x27e3f21 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0xf55>
 4fd2a5e:      	movl	$0x33, %esi
 4fd2a63:      	callq	*0x16e417(%rip)         # 0x5140e80 <writev+0x5140e80>
 4fd2a69:      	int3
 4fd2a6a:      	int3
 4fd2a6b:      	int3
 4fd2a6c:      	int3
 4fd2a6d:      	int3
 4fd2a6e:      	int3
 4fd2a6f:      	int3

0000000004fd3670 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq>:
 4fd3670:      	pushq	%rbp
 4fd3671:      	movq	%rsp, %rbp
 4fd3674:      	pushq	%r15
 4fd3676:      	pushq	%r14
 4fd3678:      	pushq	%r13
 4fd367a:      	pushq	%r12
 4fd367c:      	pushq	%rbx
 4fd367d:      	subq	$0x318, %rsp            # imm = 0x318
 4fd3684:      	movq	%rdx, %r15
 4fd3687:      	movq	%rsi, %r14
 4fd368a:      	movq	%rdi, %rbx
 4fd368d:      	leaq	-0x338(%rbp), %r12
 4fd3694:      	xorl	%r13d, %r13d
 4fd3697:      	movl	$0x30d, %edx            # imm = 0x30D
 4fd369c:      	movq	%r12, %rdi
 4fd369f:      	xorl	%esi, %esi
 4fd36a1:      	callq	*0x16d829(%rip)         # 0x5140ed0 <writev+0x5140ed0>
 4fd36a7:      	testq	%r15, %r15
 4fd36aa:      	je	0x4fd3a3d <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3cd>
 4fd36b0:      	xorl	%r13d, %r13d
 4fd36b3:      	xorl	%esi, %esi
 4fd36b5:      	movq	%r15, %rcx
 4fd36b8:      	nopl	(%rax,%rax)
 4fd36c0:      	movzbl	(%r14,%rsi), %r8d
 4fd36c5:      	cmpb	$0x30, %r8b
 4fd36c9:      	jne	0x4fd36db <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x6b>
 4fd36cb:      	decq	%rcx
 4fd36ce:      	incq	%rsi
 4fd36d1:      	cmpq	%rsi, %r15
 4fd36d4:      	jne	0x4fd36c0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x50>
 4fd36d6:      	jmp	0x4fd3a3d <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3cd>
 4fd36db:      	leal	-0x30(%r8), %edx
 4fd36df:      	cmpb	$0x9, %dl
 4fd36e2:      	ja	0x4fd3775 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x105>
 4fd36e8:      	movq	%rsi, %rax
 4fd36eb:      	notq	%rax
 4fd36ee:      	addq	%r15, %rax
 4fd36f1:      	xorl	%r13d, %r13d
 4fd36f4:      	nopw	%cs:(%rax,%rax)
 4fd3700:      	cmpq	$0x2ff, %r13            # imm = 0x2FF
 4fd3707:      	ja	0x4fd3711 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0xa1>
 4fd3709:      	movb	%dl, -0x330(%rbp,%r13)
 4fd3711:      	cmpq	%r13, %rax
 4fd3714:      	je	0x4fd37dc <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x16c>
 4fd371a:      	leaq	(%r14,%r13), %rdx
 4fd371e:      	movzbl	0x1(%rdx,%rsi), %r8d
 4fd3724:      	leal	-0x30(%r8), %edx
 4fd3728:      	incq	%r13
 4fd372b:      	cmpb	$0x9, %dl
 4fd372e:      	jbe	0x4fd3700 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x90>
 4fd3730:      	movq	%r13, -0x338(%rbp)
 4fd3737:      	movq	%rcx, %rax
 4fd373a:      	subq	%r13, %rax
 4fd373d:      	addq	%r14, %rsi
 4fd3740:      	leaq	(%rsi,%r13), %rdi
 4fd3744:      	cmpb	$0x2e, %r8b
 4fd3748:      	jne	0x4fd37ef <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x17f>
 4fd374e:      	movq	%r13, %rdx
 4fd3751:      	notq	%rdx
 4fd3754:      	addq	%rcx, %rdx
 4fd3757:      	leaq	(%rsi,%r13), %rcx
 4fd375b:      	incq	%rcx
 4fd375e:      	testq	%r13, %r13
 4fd3761:      	je	0x4fd37a4 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x134>
 4fd3763:      	movq	%rdx, %rax
 4fd3766:      	cmpq	$0x8, %rax
 4fd376a:      	jae	0x4fd3896 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x226>
 4fd3770:      	jmp	0x4fd3914 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x2a4>
 4fd3775:      	leaq	(%r14,%rsi), %rdi
 4fd3779:      	movq	%r15, %rax
 4fd377c:      	subq	%rsi, %rax
 4fd377f:      	movq	$0x0, -0x338(%rbp)
 4fd378a:      	cmpb	$0x2e, %r8b
 4fd378e:      	jne	0x4fd384a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x1da>
 4fd3794:      	leaq	(%r14,%rsi), %rcx
 4fd3798:      	incq	%rcx
 4fd379b:      	notq	%rsi
 4fd379e:      	addq	%r15, %rsi
 4fd37a1:      	movq	%rsi, %rdx
 4fd37a4:      	testq	%rdx, %rdx
 4fd37a7:      	je	0x4fd387a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x20a>
 4fd37ad:      	addq	%rax, %rdi
 4fd37b0:      	xorl	%r13d, %r13d
 4fd37b3:      	xorl	%esi, %esi
 4fd37b5:      	nopw	%cs:(%rax,%rax)
 4fd37c0:      	cmpb	$0x30, (%rcx,%rsi)
 4fd37c4:      	jne	0x4fd3884 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x214>
 4fd37ca:      	incq	%rsi
 4fd37cd:      	cmpq	%rsi, %rdx
 4fd37d0:      	jne	0x4fd37c0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x150>
 4fd37d2:      	xorl	%eax, %eax
 4fd37d4:      	movq	%rdi, %rcx
 4fd37d7:      	jmp	0x4fd398a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x31a>
 4fd37dc:      	addq	%r14, %rsi
 4fd37df:      	leaq	(%rsi,%r13), %rdi
 4fd37e3:      	incq	%rdi
 4fd37e6:      	incq	%r13
 4fd37e9:      	movq	%r13, (%r12)
 4fd37ed:      	xorl	%eax, %eax
 4fd37ef:      	testq	%r13, %r13
 4fd37f2:      	je	0x4fd399d <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x32d>
 4fd37f8:      	movq	%r15, %rsi
 4fd37fb:      	subq	%rax, %rsi
 4fd37fe:      	jb	0x4fd3a82 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x412>
 4fd3804:      	jne	0x4fd385b <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x1eb>
 4fd3806:      	xorl	%ecx, %ecx
 4fd3808:      	movl	-0x30(%rbp), %edx
 4fd380b:      	addl	%ecx, %edx
 4fd380d:      	subq	%rcx, %r13
 4fd3810:      	movq	%r13, -0x338(%rbp)
 4fd3817:      	leal	(%rdx,%r13), %ecx
 4fd381b:      	movl	%ecx, -0x30(%rbp)
 4fd381e:      	cmpq	$0x301, %r13            # imm = 0x301
 4fd3825:      	jb	0x4fd384d <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x1dd>
 4fd3827:      	movb	$0x1, -0x2c(%rbp)
 4fd382b:      	movq	$0x300, -0x338(%rbp)    # imm = 0x300
 4fd3836:      	movl	$0x300, %r13d           # imm = 0x300
 4fd383c:      	testq	%rax, %rax
 4fd383f:      	jne	0x4fd39a9 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x339>
 4fd3845:      	jmp	0x4fd3a37 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3c7>
 4fd384a:      	xorl	%r13d, %r13d
 4fd384d:      	testq	%rax, %rax
 4fd3850:      	jne	0x4fd39a9 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x339>
 4fd3856:      	jmp	0x4fd3a37 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3c7>
 4fd385b:      	xorl	%ecx, %ecx
 4fd385d:      	jmp	0x4fd3865 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x1f5>
 4fd385f:      	nop
 4fd3860:      	decq	%rsi
 4fd3863:      	je	0x4fd3808 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x198>
 4fd3865:      	movzbl	-0x1(%r14,%rsi), %edx
 4fd386b:      	cmpl	$0x2e, %edx
 4fd386e:      	je	0x4fd3860 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x1f0>
 4fd3870:      	cmpl	$0x30, %edx
 4fd3873:      	jne	0x4fd3808 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x198>
 4fd3875:      	incq	%rcx
 4fd3878:      	jmp	0x4fd3860 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x1f0>
 4fd387a:      	xorl	%r13d, %r13d
 4fd387d:      	xorl	%eax, %eax
 4fd387f:      	jmp	0x4fd398a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x31a>
 4fd3884:      	addq	%rsi, %rcx
 4fd3887:      	movq	%rdx, %rax
 4fd388a:      	subq	%rsi, %rax
 4fd388d:      	xorl	%r13d, %r13d
 4fd3890:      	cmpq	$0x8, %rax
 4fd3894:      	jb	0x4fd3914 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x2a4>
 4fd3896:      	addq	$0x8, %r13
 4fd389a:      	movabsq	$0x4646464646464646, %rsi # imm = 0x4646464646464646
 4fd38a4:      	movabsq	$-0x3030303030303030, %r8 # imm = 0xCFCFCFCFCFCFCFD0
 4fd38ae:      	movabsq	$-0x7f7f7f7f7f7f7f80, %r9 # imm = 0x8080808080808080
 4fd38b8:      	movq	%r13, %rdi
 4fd38bb:      	nopl	(%rax,%rax)
 4fd38c0:      	leaq	-0x8(%rdi), %r13
 4fd38c4:      	cmpq	$0x300, %rdi            # imm = 0x300
 4fd38cb:      	jae	0x4fd3919 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x2a9>
 4fd38cd:      	movq	(%rcx), %r10
 4fd38d0:      	leaq	(%r10,%rsi), %r11
 4fd38d4:      	addq	%r8, %r10
 4fd38d7:      	orq	%r10, %r11
 4fd38da:      	testq	%r9, %r11
 4fd38dd:      	jne	0x4fd3919 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x2a9>
 4fd38df:      	cmpq	$0x300, %r13            # imm = 0x300
 4fd38e6:      	ja	0x4fd3a94 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x424>
 4fd38ec:      	movq	%r10, -0x338(%rbp,%rdi)
 4fd38f4:      	movq	%rdi, -0x338(%rbp)
 4fd38fb:      	addq	$-0x8, %rax
 4fd38ff:      	addq	$0x8, %rcx
 4fd3903:      	addq	$0x8, %rdi
 4fd3907:      	cmpq	$0x7, %rax
 4fd390b:      	ja	0x4fd38c0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x250>
 4fd390d:      	addq	$-0x8, %rdi
 4fd3911:      	movq	%rdi, %r13
 4fd3914:      	testq	%rax, %rax
 4fd3917:      	je	0x4fd3971 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x301>
 4fd3919:      	movzbl	(%rcx), %esi
 4fd391c:      	addb	$-0x30, %sil
 4fd3920:      	cmpb	$0x9, %sil
 4fd3924:      	ja	0x4fd3983 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x313>
 4fd3926:      	leaq	-0x1(%rax), %rdi
 4fd392a:      	leaq	(%r12,%r13), %r8
 4fd392e:      	addq	$0x8, %r8
 4fd3932:      	xorl	%r10d, %r10d
 4fd3935:      	nopw	%cs:(%rax,%rax)
 4fd3940:      	movq	%r10, %r9
 4fd3943:      	addq	%r13, %r10
 4fd3946:      	cmpq	$0x2ff, %r10            # imm = 0x2FF
 4fd394d:      	ja	0x4fd3953 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x2e3>
 4fd394f:      	movb	%sil, (%r8,%r9)
 4fd3953:      	cmpq	%r9, %rdi
 4fd3956:      	je	0x4fd3975 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x305>
 4fd3958:      	movzbl	0x1(%rcx,%r9), %esi
 4fd395e:      	addb	$-0x30, %sil
 4fd3962:      	leaq	0x1(%r9), %r10
 4fd3966:      	decq	%rax
 4fd3969:      	cmpb	$0x9, %sil
 4fd396d:      	jbe	0x4fd3940 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x2d0>
 4fd396f:      	jmp	0x4fd3977 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x307>
 4fd3971:      	xorl	%eax, %eax
 4fd3973:      	jmp	0x4fd398a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x31a>
 4fd3975:      	xorl	%eax, %eax
 4fd3977:      	addq	%r9, %rcx
 4fd397a:      	incq	%rcx
 4fd397d:      	addq	%r9, %r13
 4fd3980:      	incq	%r13
 4fd3983:      	movq	%r13, -0x338(%rbp)
 4fd398a:      	movl	%eax, %esi
 4fd398c:      	subl	%edx, %esi
 4fd398e:      	movl	%esi, -0x30(%rbp)
 4fd3991:      	movq	%rcx, %rdi
 4fd3994:      	testq	%r13, %r13
 4fd3997:      	jne	0x4fd37f8 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x188>
 4fd399d:      	xorl	%r13d, %r13d
 4fd39a0:      	testq	%rax, %rax
 4fd39a3:      	je	0x4fd3a37 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3c7>
 4fd39a9:      	movzbl	(%rdi), %ecx
 4fd39ac:      	orl	$0x20, %ecx
 4fd39af:      	cmpl	$0x65, %ecx
 4fd39b2:      	jne	0x4fd3a37 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3c7>
 4fd39b8:      	movq	%rax, %rdx
 4fd39bb:      	decq	%rdx
 4fd39be:      	je	0x4fd3a24 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3b4>
 4fd39c0:      	leaq	0x1(%rdi), %rsi
 4fd39c4:      	movzbl	(%rsi), %ecx
 4fd39c7:      	cmpl	$0x2d, %ecx
 4fd39ca:      	je	0x4fd39d1 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x361>
 4fd39cc:      	cmpl	$0x2b, %ecx
 4fd39cf:      	jne	0x4fd39e1 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x371>
 4fd39d1:      	addq	$-0x2, %rax
 4fd39d5:      	je	0x4fd3a28 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3b8>
 4fd39d7:      	addq	$0x2, %rdi
 4fd39db:      	movq	%rdi, %rsi
 4fd39de:      	movq	%rax, %rdx
 4fd39e1:      	xorl	%eax, %eax
 4fd39e3:      	xorl	%edi, %edi
 4fd39e5:      	xorl	%r8d, %r8d
 4fd39e8:      	nopl	(%rax,%rax)
 4fd39f0:      	movzbl	(%rsi,%rdi), %r9d
 4fd39f5:      	addb	$-0x30, %r9b
 4fd39f9:      	cmpb	$0x9, %r9b
 4fd39fd:      	ja	0x4fd3a2a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3ba>
 4fd39ff:      	cmpl	$0x10000, %r8d          # imm = 0x10000
 4fd3a06:      	leal	(%r8,%r8,4), %r10d
 4fd3a0a:      	movzbl	%r9b, %r9d
 4fd3a0e:      	leal	(%r9,%r10,2), %r9d
 4fd3a12:      	cmovll	%r9d, %eax
 4fd3a16:      	cmovll	%r9d, %r8d
 4fd3a1a:      	incq	%rdi
 4fd3a1d:      	cmpq	%rdi, %rdx
 4fd3a20:      	jne	0x4fd39f0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x380>
 4fd3a22:      	jmp	0x4fd3a2a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3ba>
 4fd3a24:      	xorl	%edx, %edx
 4fd3a26:      	jmp	0x4fd3a34 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3c4>
 4fd3a28:      	xorl	%eax, %eax
 4fd3a2a:      	movl	%eax, %edx
 4fd3a2c:      	negl	%edx
 4fd3a2e:      	cmpb	$0x2d, %cl
 4fd3a31:      	cmovnel	%eax, %edx
 4fd3a34:      	addl	%edx, -0x30(%rbp)
 4fd3a37:      	cmpq	$0x12, %r13
 4fd3a3b:      	ja	0x4fd3a58 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt11decimal_seq17parse_decimal_seq+0x3e8>
 4fd3a3d:      	leaq	-0x330(,%r13), %rdi
 4fd3a45:      	addq	%rbp, %rdi
 4fd3a48:      	movl	$0x13, %edx
 4fd3a4d:      	subq	%r13, %rdx
 4fd3a50:      	xorl	%esi, %esi
 4fd3a52:      	callq	*0x16d478(%rip)         # 0x5140ed0 <writev+0x5140ed0>
 4fd3a58:      	leaq	-0x338(%rbp), %rsi
 4fd3a5f:      	movl	$0x310, %edx            # imm = 0x310
 4fd3a64:      	movq	%rbx, %rdi
 4fd3a67:      	callq	*0x16d2cb(%rip)         # 0x5140d38 <writev+0x5140d38>
 4fd3a6d:      	movq	%rbx, %rax
 4fd3a70:      	addq	$0x318, %rsp            # imm = 0x318
 4fd3a77:      	popq	%rbx
 4fd3a78:      	popq	%r12
 4fd3a7a:      	popq	%r13
 4fd3a7c:      	popq	%r14
 4fd3a7e:      	popq	%r15
 4fd3a80:      	popq	%rbp
 4fd3a81:      	retq
 4fd3a82:      	leaq	0x16ca1f(%rip), %rcx    # 0x51404a8 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data11conversions13UPPERCASE_LUT+0x58>
 4fd3a89:      	xorl	%edi, %edi
 4fd3a8b:      	movq	%r15, %rdx
 4fd3a8e:      	callq	*0x16d3dc(%rip)         # 0x5140e70 <writev+0x5140e70>
 4fd3a94:      	addq	$-0x8, %rdi
 4fd3a98:      	leaq	0x16c9f1(%rip), %rcx    # 0x5140490 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data11conversions13UPPERCASE_LUT+0x40>
 4fd3a9f:      	movl	$0x300, %esi            # imm = 0x300
 4fd3aa4:      	movl	$0x300, %edx            # imm = 0x300
 4fd3aa9:      	callq	*0x16d3c1(%rip)         # 0x5140e70 <writev+0x5140e70>
 4fd3aaf:      	int3

0000000004fd3ab0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number>:
 4fd3ab0:      	pushq	%rbp
 4fd3ab1:      	movq	%rsp, %rbp
 4fd3ab4:      	pushq	%r15
 4fd3ab6:      	pushq	%r14
 4fd3ab8:      	pushq	%r13
 4fd3aba:      	pushq	%r12
 4fd3abc:      	pushq	%rbx
 4fd3abd:      	subq	$0x18, %rsp
 4fd3ac1:      	movq	%rdi, %r13
 4fd3ac4:      	cmpq	$0x8, %rdx
 4fd3ac8:      	jb	0x4fd3b79 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0xc9>
 4fd3ace:      	movabsq	$-0x3030303030303030, %rcx # imm = 0xCFCFCFCFCFCFCFD0
 4fd3ad8:      	movabsq	$-0x7f7f7f7f7f7f7f80, %r9 # imm = 0x8080808080808080
 4fd3ae2:      	movabsq	$0x4646464646464646, %r10 # imm = 0x4646464646464646
 4fd3aec:      	movabsq	$0xf424000000064, %rbx  # imm = 0xF424000000064
 4fd3af6:      	movabsq	$0x271000000001, %r14   # imm = 0x271000000001
 4fd3b00:      	movabsq	$0xff000000ff, %r15     # imm = 0xFF000000FF
 4fd3b0a:      	xorl	%eax, %eax
 4fd3b0c:      	movq	%rsi, %r8
 4fd3b0f:      	movq	%rdx, %r11
 4fd3b12:      	nopw	%cs:(%rax,%rax)
 4fd3b20:      	movq	(%r8), %rdi
 4fd3b23:      	leaq	(%rdi,%r10), %r12
 4fd3b27:      	addq	%rcx, %rdi
 4fd3b2a:      	orq	%rdi, %r12
 4fd3b2d:      	testq	%r9, %r12
 4fd3b30:      	jne	0x4fd3b86 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0xd6>
 4fd3b32:      	imulq	$0x5f5e100, %rax, %r12  # imm = 0x5F5E100
 4fd3b39:      	leaq	(%rdi,%rdi,4), %rax
 4fd3b3d:      	shrq	$0x8, %rdi
 4fd3b41:      	leaq	(%rdi,%rax,2), %rax
 4fd3b45:      	movq	%rax, %rdi
 4fd3b48:      	andq	%r15, %rdi
 4fd3b4b:      	imulq	%rbx, %rdi
 4fd3b4f:      	shrq	$0x10, %rax
 4fd3b53:      	andq	%r15, %rax
 4fd3b56:      	imulq	%r14, %rax
 4fd3b5a:      	addq	%rdi, %rax
 4fd3b5d:      	shrq	$0x20, %rax
 4fd3b61:      	addq	%r12, %rax
 4fd3b64:      	addq	$-0x8, %r11
 4fd3b68:      	addq	$0x8, %r8
 4fd3b6c:      	cmpq	$0x7, %r11
 4fd3b70:      	ja	0x4fd3b20 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x70>
 4fd3b72:      	testq	%r11, %r11
 4fd3b75:      	jne	0x4fd3b86 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0xd6>
 4fd3b77:      	jmp	0x4fd3bb2 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x102>
 4fd3b79:      	xorl	%eax, %eax
 4fd3b7b:      	movq	%rdx, %r11
 4fd3b7e:      	movq	%rsi, %r8
 4fd3b81:      	testq	%r11, %r11
 4fd3b84:      	je	0x4fd3bb2 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x102>
 4fd3b86:      	xorl	%ebx, %ebx
 4fd3b88:      	nopl	(%rax,%rax)
 4fd3b90:      	movzbl	(%r8,%rbx), %ecx
 4fd3b95:      	leal	-0x30(%rcx), %edi
 4fd3b98:      	cmpb	$0x9, %dil
 4fd3b9c:      	ja	0x4fd3bd9 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x129>
 4fd3b9e:      	leaq	(%rax,%rax,4), %rax
 4fd3ba2:      	movzbl	%dil, %ecx
 4fd3ba6:      	leaq	(%rcx,%rax,2), %rax
 4fd3baa:      	incq	%rbx
 4fd3bad:      	cmpq	%rbx, %r11
 4fd3bb0:      	jne	0x4fd3b90 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0xe0>
 4fd3bb2:      	testq	%rdx, %rdx
 4fd3bb5:      	je	0x4fd3cbc <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x20c>
 4fd3bbb:      	movb	$0x1, %r8b
 4fd3bbe:      	xorl	%r12d, %r12d
 4fd3bc1:      	movq	%rdx, %r9
 4fd3bc4:      	xorl	%r11d, %r11d
 4fd3bc7:      	xorl	%r10d, %r10d
 4fd3bca:      	cmpq	$0x14, %r9
 4fd3bce:      	jl	0x4fd3db4 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x304>
 4fd3bd4:      	jmp	0x4fd3e35 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x385>
 4fd3bd9:      	movq	%r11, %r12
 4fd3bdc:      	subq	%rbx, %r12
 4fd3bdf:      	cmpb	$0x2e, %cl
 4fd3be2:      	jne	0x4fd3cc4 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x214>
 4fd3be8:      	movq	%r13, -0x40(%rbp)
 4fd3bec:      	movq	%rbx, %rcx
 4fd3bef:      	notq	%rcx
 4fd3bf2:      	movq	%rcx, -0x30(%rbp)
 4fd3bf6:      	leaq	(%rcx,%r11), %r14
 4fd3bfa:      	addq	%rbx, %r8
 4fd3bfd:      	incq	%r8
 4fd3c00:      	movq	%r12, -0x38(%rbp)
 4fd3c04:      	cmpq	$0x9, %r12
 4fd3c08:      	jl	0x4fd3ca6 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x1f6>
 4fd3c0e:      	movabsq	$-0x7f7f7f7f7f7f7f80, %r10 # imm = 0x8080808080808080
 4fd3c18:      	movabsq	$0x4646464646464646, %r12 # imm = 0x4646464646464646
 4fd3c22:      	movabsq	$0x271000000001, %rcx   # imm = 0x271000000001
 4fd3c2c:      	movabsq	$0xff000000ff, %r15     # imm = 0xFF000000FF
 4fd3c36:      	nopw	%cs:(%rax,%rax)
 4fd3c40:      	movq	(%r8), %rdi
 4fd3c43:      	leaq	(%rdi,%r12), %r9
 4fd3c47:      	movabsq	$-0x3030303030303030, %r13 # imm = 0xCFCFCFCFCFCFCFD0
 4fd3c51:      	addq	%r13, %rdi
 4fd3c54:      	orq	%rdi, %r9
 4fd3c57:      	testq	%r10, %r9
 4fd3c5a:      	jne	0x4fd3cd2 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x222>
 4fd3c5c:      	imulq	$0x5f5e100, %rax, %r9   # imm = 0x5F5E100
 4fd3c63:      	leaq	(%rdi,%rdi,4), %rax
 4fd3c67:      	shrq	$0x8, %rdi
 4fd3c6b:      	leaq	(%rdi,%rax,2), %rax
 4fd3c6f:      	movq	%rax, %rdi
 4fd3c72:      	andq	%r15, %rdi
 4fd3c75:      	movabsq	$0xf424000000064, %r13  # imm = 0xF424000000064
 4fd3c7f:      	imulq	%r13, %rdi
 4fd3c83:      	shrq	$0x10, %rax
 4fd3c87:      	andq	%r15, %rax
 4fd3c8a:      	imulq	%rcx, %rax
 4fd3c8e:      	addq	%rdi, %rax
 4fd3c91:      	shrq	$0x20, %rax
 4fd3c95:      	addq	%r9, %rax
 4fd3c98:      	addq	$-0x8, %r14
 4fd3c9c:      	addq	$0x8, %r8
 4fd3ca0:      	cmpq	$0x7, %r14
 4fd3ca4:      	ja	0x4fd3c40 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x190>
 4fd3ca6:      	testq	%r14, %r14
 4fd3ca9:      	movq	-0x40(%rbp), %r13
 4fd3cad:      	jne	0x4fd3cd6 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x226>
 4fd3caf:      	xorl	%r14d, %r14d
 4fd3cb2:      	movq	-0x38(%rbp), %r12
 4fd3cb6:      	movq	-0x30(%rbp), %r15
 4fd3cba:      	jmp	0x4fd3d19 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x269>
 4fd3cbc:      	movb	$0x2, %r11b
 4fd3cbf:      	jmp	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3cc4:      	addq	%rbx, %r8
 4fd3cc7:      	xorl	%r15d, %r15d
 4fd3cca:      	movq	%r12, %r14
 4fd3ccd:      	xorl	%r10d, %r10d
 4fd3cd0:      	jmp	0x4fd3d2f <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x27f>
 4fd3cd2:      	movq	-0x40(%rbp), %r13
 4fd3cd6:      	movq	-0x38(%rbp), %r12
 4fd3cda:      	movq	-0x30(%rbp), %r15
 4fd3cde:      	movq	%r8, %rcx
 4fd3ce1:      	addq	%r14, %r8
 4fd3ce4:      	nopw	%cs:(%rax,%rax)
 4fd3cf0:      	movzbl	(%rcx), %edi
 4fd3cf3:      	addb	$-0x30, %dil
 4fd3cf7:      	cmpb	$0x9, %dil
 4fd3cfb:      	ja	0x4fd3d16 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x266>
 4fd3cfd:      	incq	%rcx
 4fd3d00:      	leaq	(%rax,%rax,4), %rax
 4fd3d04:      	movzbl	%dil, %edi
 4fd3d08:      	leaq	(%rdi,%rax,2), %rax
 4fd3d0c:      	decq	%r14
 4fd3d0f:      	jne	0x4fd3cf0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x240>
 4fd3d11:      	xorl	%r14d, %r14d
 4fd3d14:      	jmp	0x4fd3d19 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x269>
 4fd3d16:      	movq	%rcx, %r8
 4fd3d19:      	movq	%r11, %rcx
 4fd3d1c:      	subq	%r14, %rcx
 4fd3d1f:      	addq	%rcx, %r15
 4fd3d22:      	movq	%r14, %rcx
 4fd3d25:      	subq	%r11, %rcx
 4fd3d28:      	leaq	(%rcx,%rbx), %r10
 4fd3d2c:      	incq	%r10
 4fd3d2f:      	movq	%rdx, %r9
 4fd3d32:      	subq	%r11, %r9
 4fd3d35:      	addq	%r15, %r9
 4fd3d38:      	movb	$0x2, %r11b
 4fd3d3b:      	addq	%rbx, %r9
 4fd3d3e:      	je	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3d44:      	testq	%r14, %r14
 4fd3d47:      	je	0x4fd3d93 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x2e3>
 4fd3d49:      	movzbl	(%r8), %ecx
 4fd3d4d:      	orl	$0x20, %ecx
 4fd3d50:      	cmpl	$0x65, %ecx
 4fd3d53:      	jne	0x4fd3da4 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x2f4>
 4fd3d55:      	movq	%r14, %r15
 4fd3d58:      	decq	%r15
 4fd3d5b:      	je	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3d61:      	movzbl	0x1(%r8), %ebx
 4fd3d66:      	cmpl	$0x2d, %ebx
 4fd3d69:      	je	0x4fd3d70 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x2c0>
 4fd3d6b:      	cmpl	$0x2b, %ebx
 4fd3d6e:      	jne	0x4fd3dc7 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x317>
 4fd3d70:      	addq	$-0x2, %r14
 4fd3d74:      	je	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3d7a:      	movzbl	0x2(%r8), %ecx
 4fd3d7f:      	addq	$0x2, %r8
 4fd3d83:      	movq	%r14, %r15
 4fd3d86:      	addb	$-0x30, %cl
 4fd3d89:      	cmpb	$0x9, %cl
 4fd3d8c:      	jbe	0x4fd3dd8 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x328>
 4fd3d8e:      	jmp	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3d93:      	movb	$0x1, %r8b
 4fd3d96:      	xorl	%r11d, %r11d
 4fd3d99:      	cmpq	$0x14, %r9
 4fd3d9d:      	jl	0x4fd3db4 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x304>
 4fd3d9f:      	jmp	0x4fd3e35 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x385>
 4fd3da4:      	xorl	%r8d, %r8d
 4fd3da7:      	xorl	%r11d, %r11d
 4fd3daa:      	cmpq	$0x14, %r9
 4fd3dae:      	jge	0x4fd3e35 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x385>
 4fd3db4:      	xorl	%ecx, %ecx
 4fd3db6:      	movb	$0x2, %r11b
 4fd3db9:      	testb	%r8b, %r8b
 4fd3dbc:      	jne	0x4fd3f4a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x49a>
 4fd3dc2:      	jmp	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3dc7:      	incq	%r8
 4fd3dca:      	movl	%ebx, %ecx
 4fd3dcc:      	addb	$-0x30, %cl
 4fd3dcf:      	cmpb	$0x9, %cl
 4fd3dd2:      	ja	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3dd8:      	xorl	%ecx, %ecx
 4fd3dda:      	xorl	%edi, %edi
 4fd3ddc:      	nopl	(%rax)
 4fd3de0:      	movzbl	(%r8), %r11d
 4fd3de4:      	addb	$-0x30, %r11b
 4fd3de8:      	cmpb	$0x9, %r11b
 4fd3dec:      	ja	0x4fd3e14 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x364>
 4fd3dee:      	incq	%r8
 4fd3df1:      	cmpq	$0x10000, %rdi          # imm = 0x10000
 4fd3df8:      	leaq	(%rdi,%rdi,4), %r14
 4fd3dfc:      	movzbl	%r11b, %r11d
 4fd3e00:      	leaq	(%r11,%r14,2), %r11
 4fd3e04:      	cmovlq	%r11, %rcx
 4fd3e08:      	cmovlq	%r11, %rdi
 4fd3e0c:      	decq	%r15
 4fd3e0f:      	jne	0x4fd3de0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x330>
 4fd3e11:      	xorl	%r15d, %r15d
 4fd3e14:      	movq	%rcx, %r11
 4fd3e17:      	negq	%r11
 4fd3e1a:      	cmpb	$0x2d, %bl
 4fd3e1d:      	cmovneq	%rcx, %r11
 4fd3e21:      	addq	%r11, %r10
 4fd3e24:      	testq	%r15, %r15
 4fd3e27:      	sete	%r8b
 4fd3e2b:      	cmpq	$0x14, %r9
 4fd3e2f:      	jl	0x4fd3db4 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x304>
 4fd3e35:      	testq	%rdx, %rdx
 4fd3e38:      	je	0x4fd3f70 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4c0>
 4fd3e3e:      	addq	$-0x13, %r9
 4fd3e42:      	xorl	%ecx, %ecx
 4fd3e44:      	xorl	%edi, %edi
 4fd3e46:      	jmp	0x4fd3e67 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x3b7>
 4fd3e48:      	nopl	(%rax,%rax)
 4fd3e50:      	subb	$0x2f, %bl
 4fd3e53:      	movzbl	%bl, %ebx
 4fd3e56:      	cmovbl	%ecx, %ebx
 4fd3e59:      	movzbl	%bl, %ebx
 4fd3e5c:      	subq	%rbx, %r9
 4fd3e5f:      	incq	%rdi
 4fd3e62:      	cmpq	%rdi, %rdx
 4fd3e65:      	je	0x4fd3e75 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x3c5>
 4fd3e67:      	movzbl	(%rsi,%rdi), %ebx
 4fd3e6b:      	cmpl	$0x2e, %ebx
 4fd3e6e:      	je	0x4fd3e50 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x3a0>
 4fd3e70:      	cmpl	$0x30, %ebx
 4fd3e73:      	je	0x4fd3e50 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x3a0>
 4fd3e75:      	testq	%r9, %r9
 4fd3e78:      	jle	0x4fd3f3c <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x48c>
 4fd3e7e:      	movabsq	$0xde0b6b3a763ffff, %rbx # imm = 0xDE0B6B3A763FFFF
 4fd3e88:      	negq	%rdx
 4fd3e8b:      	xorl	%eax, %eax
 4fd3e8d:      	nopl	(%rax)
 4fd3e90:      	movq	%rdx, %rcx
 4fd3e93:      	movzbl	(%rsi), %edx
 4fd3e96:      	addb	$-0x30, %dl
 4fd3e99:      	cmpb	$0x9, %dl
 4fd3e9c:      	ja	0x4fd3ed3 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x423>
 4fd3e9e:      	incq	%rsi
 4fd3ea1:      	leaq	(%rax,%rax,4), %rax
 4fd3ea5:      	movzbl	%dl, %edx
 4fd3ea8:      	leaq	(%rdx,%rax,2), %rax
 4fd3eac:      	movq	%rcx, %rdx
 4fd3eaf:      	incq	%rdx
 4fd3eb2:      	sete	%dil
 4fd3eb6:      	cmpq	%rbx, %rax
 4fd3eb9:      	ja	0x4fd3ec0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x410>
 4fd3ebb:      	testb	%dil, %dil
 4fd3ebe:      	je	0x4fd3e90 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x3e0>
 4fd3ec0:      	cmpq	%rbx, %rax
 4fd3ec3:      	jbe	0x4fd3f20 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x470>
 4fd3ec5:      	addq	%rdx, %r12
 4fd3ec8:      	negq	%r12
 4fd3ecb:      	addq	%r11, %r12
 4fd3ece:      	movq	%r12, %r10
 4fd3ed1:      	jmp	0x4fd3f3c <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x48c>
 4fd3ed3:      	negq	%rcx
 4fd3ed6:      	movq	%rcx, %rdx
 4fd3ed9:      	decq	%rdx
 4fd3edc:      	je	0x4fd3f2e <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x47e>
 4fd3ede:      	incq	%rsi
 4fd3ee1:      	movq	%rdx, %rcx
 4fd3ee4:      	nopw	%cs:(%rax,%rax)
 4fd3ef0:      	movzbl	(%rsi), %edi
 4fd3ef3:      	addb	$-0x30, %dil
 4fd3ef7:      	cmpb	$0x9, %dil
 4fd3efb:      	ja	0x4fd3f33 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x483>
 4fd3efd:      	leaq	-0x1(%rcx), %r10
 4fd3f01:      	leaq	(%rax,%rax,4), %rax
 4fd3f05:      	movzbl	%dil, %edi
 4fd3f09:      	leaq	(%rdi,%rax,2), %rax
 4fd3f0d:      	cmpq	%rbx, %rax
 4fd3f10:      	ja	0x4fd3f36 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x486>
 4fd3f12:      	incq	%rsi
 4fd3f15:      	cmpq	$0x1, %rcx
 4fd3f19:      	movq	%r10, %rcx
 4fd3f1c:      	jne	0x4fd3ef0 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x440>
 4fd3f1e:      	jmp	0x4fd3f36 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x486>
 4fd3f20:      	cmpq	$-0x1, %rcx
 4fd3f24:      	je	0x4fd3f70 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4c0>
 4fd3f26:      	negq	%rdx
 4fd3f29:      	decq	%rdx
 4fd3f2c:      	jne	0x4fd3ede <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x42e>
 4fd3f2e:      	xorl	%r10d, %r10d
 4fd3f31:      	jmp	0x4fd3f36 <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x486>
 4fd3f33:      	movq	%rcx, %r10
 4fd3f36:      	subq	%rdx, %r10
 4fd3f39:      	addq	%r11, %r10
 4fd3f3c:      	testq	%r9, %r9
 4fd3f3f:      	setg	%cl
 4fd3f42:      	movb	$0x2, %r11b
 4fd3f45:      	testb	%r8b, %r8b
 4fd3f48:      	je	0x4fd3f5a <_RNvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt5parse12parse_number+0x4aa>
 4fd3f4a:      	movq	%r10, (%r13)
 4fd3f4e:      	movq	%rax, 0x8(%r13)
 4fd3f52:      	movb	$0x0, 0x10(%r13)
 4fd3f57:      	movl	%ecx, %r11d
 4fd3f5a:      	movb	%r11b, 0x11(%r13)
 4fd3f5e:      	movq	%r13, %rax
 4fd3f61:      	addq	$0x18, %rsp
 4fd3f65:      	popq	%rbx
 4fd3f66:      	popq	%r12
 4fd3f68:      	popq	%r13
 4fd3f6a:      	popq	%r14
 4fd3f6c:      	popq	%r15
 4fd3f6e:      	popq	%rbp
 4fd3f6f:      	retq
 4fd3f70:      	leaq	0x16c561(%rip), %rcx    # 0x51404d8 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data11conversions13UPPERCASE_LUT+0x88>
 4fd3f77:      	movl	$0x1, %edi
 4fd3f7c:      	xorl	%esi, %esi
 4fd3f7e:      	xorl	%edx, %edx
 4fd3f80:      	callq	*0x16ceea(%rip)         # 0x5140e70 <writev+0x5140e70>
 4fd3f86:      	int3
 4fd3f87:      	int3
 4fd3f88:      	int3
 4fd3f89:      	int3
 4fd3f8a:      	int3
 4fd3f8b:      	int3
 4fd3f8c:      	int3
 4fd3f8d:      	int3
 4fd3f8e:      	int3
 4fd3f8f:      	int3

0000000004fd7d80 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str>:
 4fd7d80:      	pushq	%rbp
 4fd7d81:      	movq	%rsp, %rbp
 4fd7d84:      	pushq	%r15
 4fd7d86:      	pushq	%r14
 4fd7d88:      	pushq	%r13
 4fd7d8a:      	pushq	%r12
 4fd7d8c:      	pushq	%rbx
 4fd7d8d:      	subq	$0x38, %rsp
 4fd7d91:      	movabsq	$-0x7f7f7f7f7f7f7f80, %rbx # imm = 0x8080808080808080
 4fd7d9b:      	movabsq	$0xa0a0a0a0a0a0a0a, %r12 # imm = 0xA0A0A0A0A0A0A0A
 4fd7da5:      	movq	0x10(%rdi), %rax
 4fd7da9:      	movq	%rax, -0x50(%rbp)
 4fd7dad:      	movq	(%rdi), %rax
 4fd7db0:      	movq	%rax, -0x48(%rbp)
 4fd7db4:      	movq	0x8(%rdi), %rax
 4fd7db8:      	movq	%rax, -0x40(%rbp)
 4fd7dbc:      	movq	%rsi, -0x30(%rbp)
 4fd7dc0:      	leaq	0x8(%rsi), %rax
 4fd7dc4:      	movq	%rax, -0x58(%rbp)
 4fd7dc8:      	xorl	%r14d, %r14d
 4fd7dcb:      	xorl	%r15d, %r15d
 4fd7dce:      	xorl	%eax, %eax
 4fd7dd0:      	movq	%rdx, -0x60(%rbp)
 4fd7dd4:      	jmp	0x4fd7e24 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0xa4>
 4fd7dd6:      	nopw	%cs:(%rax,%rax)
 4fd7de0:      	cmpb	$0xa, -0x1(%rdi,%rbx)
 4fd7de5:      	sete	%al
 4fd7de8:      	subq	%r15, %rbx
 4fd7deb:      	addq	%rdi, %r15
 4fd7dee:      	movq	-0x50(%rbp), %rcx
 4fd7df2:      	movb	%al, (%rcx)
 4fd7df4:      	movq	-0x48(%rbp), %rdi
 4fd7df8:      	movq	%r15, %rsi
 4fd7dfb:      	movq	%rbx, %rdx
 4fd7dfe:      	movq	-0x40(%rbp), %rax
 4fd7e02:      	callq	*0x18(%rax)
 4fd7e05:      	movq	%r13, %r15
 4fd7e08:      	testb	%al, %al
 4fd7e0a:      	movq	-0x60(%rbp), %rdx
 4fd7e0e:      	movq	%r12, %rbx
 4fd7e11:      	movabsq	$0xa0a0a0a0a0a0a0a, %r12 # imm = 0xA0A0A0A0A0A0A0A
 4fd7e1b:      	movl	-0x34(%rbp), %eax
 4fd7e1e:      	jne	0x4fd7fe5 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x265>
 4fd7e24:      	testb	$0x1, %al
 4fd7e26:      	jne	0x4fd7fe1 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x261>
 4fd7e2c:      	cmpq	%r14, %rdx
 4fd7e2f:      	jae	0x4fd7e40 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0xc0>
 4fd7e31:      	movq	%r14, %r13
 4fd7e34:      	movq	-0x30(%rbp), %rdi
 4fd7e38:      	jmp	0x4fd7f93 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x213>
 4fd7e3d:      	nopl	(%rax)
 4fd7e40:      	movq	-0x30(%rbp), %rdi
 4fd7e44:      	jmp	0x4fd7e5c <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0xdc>
 4fd7e46:      	nopw	%cs:(%rax,%rax)
 4fd7e50:      	movq	%r13, %r14
 4fd7e53:      	cmpq	%r13, %rdx
 4fd7e56:      	jb	0x4fd7f93 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x213>
 4fd7e5c:      	movq	%rdx, %rcx
 4fd7e5f:      	subq	%r14, %rcx
 4fd7e62:      	leaq	(%rdi,%r14), %rax
 4fd7e66:      	cmpq	$0xf, %rcx
 4fd7e6a:      	ja	0x4fd7ea0 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x120>
 4fd7e6c:      	cmpq	%r14, %rdx
 4fd7e6f:      	je	0x4fd7f90 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x210>
 4fd7e75:      	xorl	%esi, %esi
 4fd7e77:      	nopw	(%rax,%rax)
 4fd7e80:      	cmpb	$0xa, (%rax,%rsi)
 4fd7e84:      	je	0x4fd7f60 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1e0>
 4fd7e8a:      	incq	%rsi
 4fd7e8d:      	cmpq	%rsi, %rcx
 4fd7e90:      	jne	0x4fd7e80 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x100>
 4fd7e92:      	jmp	0x4fd7f90 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x210>
 4fd7e97:      	nopw	(%rax,%rax)
 4fd7ea0:      	leaq	0x7(%rax), %r11
 4fd7ea4:      	andq	$-0x8, %r11
 4fd7ea8:      	subq	%rax, %r11
 4fd7eab:      	movabsq	$0x101010101010100, %r13 # imm = 0x101010101010100
 4fd7eb5:      	jne	0x4fd7f10 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x190>
 4fd7eb7:      	leaq	-0x10(%rcx), %rsi
 4fd7ebb:      	xorl	%r11d, %r11d
 4fd7ebe:      	movq	-0x58(%rbp), %rdi
 4fd7ec2:      	addq	%r14, %rdi
 4fd7ec5:      	nopw	%cs:(%rax,%rax)
 4fd7ed0:      	movq	-0x8(%rdi,%r11), %r8
 4fd7ed5:      	movq	%r8, %r9
 4fd7ed8:      	xorq	%r12, %r9
 4fd7edb:      	movq	%r13, %r10
 4fd7ede:      	subq	%r9, %r10
 4fd7ee1:      	orq	%r8, %r10
 4fd7ee4:      	movq	(%rdi,%r11), %r8
 4fd7ee8:      	xorq	%r12, %r8
 4fd7eeb:      	movq	%r13, %r9
 4fd7eee:      	subq	%r8, %r9
 4fd7ef1:      	orq	%r8, %r9
 4fd7ef4:      	andq	%rbx, %r10
 4fd7ef7:      	andq	%r9, %r10
 4fd7efa:      	cmpq	%rbx, %r10
 4fd7efd:      	jne	0x4fd7f40 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1c0>
 4fd7eff:      	addq	$0x10, %r11
 4fd7f03:      	cmpq	%rsi, %r11
 4fd7f06:      	jbe	0x4fd7ed0 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x150>
 4fd7f08:      	jmp	0x4fd7f40 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1c0>
 4fd7f0a:      	nopw	(%rax,%rax)
 4fd7f10:      	xorl	%esi, %esi
 4fd7f12:      	nopw	%cs:(%rax,%rax)
 4fd7f20:      	cmpb	$0xa, (%rax,%rsi)
 4fd7f24:      	je	0x4fd7f60 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1e0>
 4fd7f26:      	incq	%rsi
 4fd7f29:      	cmpq	%rsi, %r11
 4fd7f2c:      	jne	0x4fd7f20 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1a0>
 4fd7f2e:      	leaq	-0x10(%rcx), %rsi
 4fd7f32:      	cmpq	%rsi, %r11
 4fd7f35:      	jbe	0x4fd7ebe <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x13e>
 4fd7f37:      	nopw	(%rax,%rax)
 4fd7f40:      	movq	%r11, %rsi
 4fd7f43:      	cmpq	%r11, %rcx
 4fd7f46:      	movq	-0x30(%rbp), %rdi
 4fd7f4a:      	je	0x4fd7f90 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x210>
 4fd7f4c:      	nopl	(%rax)
 4fd7f50:      	cmpb	$0xa, (%rax,%rsi)
 4fd7f54:      	je	0x4fd7f60 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1e0>
 4fd7f56:      	incq	%rsi
 4fd7f59:      	cmpq	%rsi, %rcx
 4fd7f5c:      	jne	0x4fd7f50 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x1d0>
 4fd7f5e:      	jmp	0x4fd7f90 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x210>
 4fd7f60:      	leaq	(%r14,%rsi), %r13
 4fd7f64:      	incq	%r13
 4fd7f67:      	addq	%rsi, %r14
 4fd7f6a:      	cmpq	%rdx, %r14
 4fd7f6d:      	jae	0x4fd7e50 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0xd0>
 4fd7f73:      	cmpb	$0xa, (%rsi,%rax)
 4fd7f77:      	jne	0x4fd7e50 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0xd0>
 4fd7f7d:      	movq	%rbx, %r12
 4fd7f80:      	xorl	%eax, %eax
 4fd7f82:      	movq	%r13, %r14
 4fd7f85:      	movq	%r13, %rbx
 4fd7f88:      	jmp	0x4fd7fa6 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x226>
 4fd7f8a:      	nopw	(%rax,%rax)
 4fd7f90:      	movq	%rdx, %r13
 4fd7f93:      	movq	%rbx, %r12
 4fd7f96:      	movb	$0x1, %al
 4fd7f98:      	movq	%r13, %r14
 4fd7f9b:      	movq	%r15, %r13
 4fd7f9e:      	movq	%rdx, %rbx
 4fd7fa1:      	cmpq	%r15, %rdx
 4fd7fa4:      	je	0x4fd7fe1 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x261>
 4fd7fa6:      	movl	%eax, -0x34(%rbp)
 4fd7fa9:      	movq	-0x50(%rbp), %rax
 4fd7fad:      	cmpb	$0x0, (%rax)
 4fd7fb0:      	je	0x4fd7fd1 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x251>
 4fd7fb2:      	movl	$0x4, %edx
 4fd7fb7:      	movq	-0x48(%rbp), %rdi
 4fd7fbb:      	leaq	-0x4c47732(%rip), %rsi  # 0x390890 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2438.llvm.6417388326552730234+0x18>
 4fd7fc2:      	movq	-0x40(%rbp), %rax
 4fd7fc6:      	callq	*0x18(%rax)
 4fd7fc9:      	movq	-0x30(%rbp), %rdi
 4fd7fcd:      	testb	%al, %al
 4fd7fcf:      	jne	0x4fd7fe5 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x265>
 4fd7fd1:      	cmpq	%r15, %rbx
 4fd7fd4:      	jne	0x4fd7de0 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x60>
 4fd7fda:      	xorl	%eax, %eax
 4fd7fdc:      	jmp	0x4fd7de8 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x68>
 4fd7fe1:      	xorl	%eax, %eax
 4fd7fe3:      	jmp	0x4fd7fe7 <_RNvXs0_NtNtCs4NRVxsYgnAr_4core3fmt8buildersNtB5_10PadAdapterNtB7_5Write9write_str+0x267>
 4fd7fe5:      	movb	$0x1, %al
 4fd7fe7:      	addq	$0x38, %rsp
 4fd7feb:      	popq	%rbx
 4fd7fec:      	popq	%r12
 4fd7fee:      	popq	%r13
 4fd7ff0:      	popq	%r14
 4fd7ff2:      	popq	%r15
 4fd7ff4:      	popq	%rbp
 4fd7ff5:      	retq
 4fd7ff6:      	int3
 4fd7ff7:      	int3
 4fd7ff8:      	int3
 4fd7ff9:      	int3
 4fd7ffa:      	int3
 4fd7ffb:      	int3
 4fd7ffc:      	int3
 4fd7ffd:      	int3
 4fd7ffe:      	int3
 4fd7fff:      	int3

0000000004fd8480 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str>:
 4fd8480:      	movl	$0x1, %ecx
 4fd8485:      	testq	%rsi, %rsi
 4fd8488:      	je	0x4fd8520 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0xa0>
 4fd848e:      	pushq	%rbp
 4fd848f:      	movq	%rsp, %rbp
 4fd8492:      	pushq	%r15
 4fd8494:      	pushq	%r14
 4fd8496:      	pushq	%r13
 4fd8498:      	pushq	%r12
 4fd849a:      	pushq	%rbx
 4fd849b:      	subq	$0x318, %rsp            # imm = 0x318
 4fd84a2:      	movq	%rsi, %r12
 4fd84a5:      	movq	%rdi, %r14
 4fd84a8:      	movzbl	(%rdi), %ebx
 4fd84ab:      	cmpl	$0x2d, %ebx
 4fd84ae:      	je	0x4fd84b5 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x35>
 4fd84b0:      	cmpl	$0x2b, %ebx
 4fd84b3:      	jne	0x4fd84c1 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x41>
 4fd84b5:      	decq	%r12
 4fd84b8:      	je	0x4fd85a3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x123>
 4fd84be:      	incq	%r14
 4fd84c1:      	leaq	-0x340(%rbp), %rdi
 4fd84c8:      	movq	%r14, %rsi
 4fd84cb:      	movq	%r12, %rdx
 4fd84ce:      	callq	*0x1740c4(%rip)         # 0x514c598 <writev+0x514c598>
 4fd84d4:      	movzbl	-0x32f(%rbp), %r15d
 4fd84dc:      	cmpb	$0x2, %r15b
 4fd84e0:      	jne	0x4fd8527 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0xa7>
 4fd84e2:      	movl	$0x1, %ecx
 4fd84e7:      	cmpq	$0x3, %r12
 4fd84eb:      	je	0x4fd85d7 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x157>
 4fd84f1:      	cmpq	$0x8, %r12
 4fd84f5:      	jne	0x4fd8602 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x182>
 4fd84fb:      	movabsq	$-0x2020202020202021, %rax # imm = 0xDFDFDFDFDFDFDFDF
 4fd8505:      	andq	(%r14), %rax
 4fd8508:      	movabsq	$0x5954494e49464e49, %rdx # imm = 0x5954494E49464E49
 4fd8512:      	cmpq	%rdx, %rax
 4fd8515:      	jne	0x4fd8602 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x182>
 4fd851b:      	jmp	0x4fd860c <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x18c>
 4fd8520:      	xorl	%edx, %edx
 4fd8522:      	jmp	0x4fd8716 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x296>
 4fd8527:      	movl	%ebx, -0x2c(%rbp)
 4fd852a:      	movq	-0x340(%rbp), %r13
 4fd8531:      	movq	-0x338(%rbp), %rbx
 4fd8538:      	leaq	-0x12(%r13), %rax
 4fd853c:      	cmpq	$-0x1c, %rax
 4fd8540:      	setb	%al
 4fd8543:      	cmpq	$0x1000001, %rbx        # imm = 0x1000001
 4fd854a:      	setae	%cl
 4fd854d:      	orb	%r15b, %cl
 4fd8550:      	orb	%al, %cl
 4fd8552:      	testb	$0x1, %cl
 4fd8555:      	je	0x4fd85ad <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x12d>
 4fd8557:      	movq	%r13, %rdi
 4fd855a:      	movq	%rbx, %rsi
 4fd855d:      	callq	0x4fcc230 <_RINvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt6lemire13compute_floatfEBa_>
 4fd8562:      	testl	%edx, %edx
 4fd8564:      	setns	%cl
 4fd8567:      	testb	%cl, %r15b
 4fd856a:      	je	0x4fd862c <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x1ac>
 4fd8570:      	incq	%rbx
 4fd8573:      	movq	%r13, %rdi
 4fd8576:      	movq	%rbx, %rsi
 4fd8579:      	movq	%rax, %r15
 4fd857c:      	movq	%rdx, %rbx
 4fd857f:      	callq	0x4fcc230 <_RINvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt6lemire13compute_floatfEBa_>
 4fd8584:      	cmpq	%rax, %r15
 4fd8587:      	sete	%al
 4fd858a:      	cmpl	%edx, %ebx
 4fd858c:      	sete	%cl
 4fd858f:      	testb	%cl, %al
 4fd8591:      	movl	-0x2c(%rbp), %r13d
 4fd8595:      	movq	%rbx, %rdx
 4fd8598:      	je	0x4fd8670 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x1f0>
 4fd859e:      	jmp	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd85a3:      	movl	$0x100, %edx            # imm = 0x100
 4fd85a8:      	jmp	0x4fd8705 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x285>
 4fd85ad:      	cmpq	$0xa, %r13
 4fd85b1:      	jg	0x4fd8639 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x1b9>
 4fd85b7:      	cvtsi2ss	%rbx, %xmm0
 4fd85bc:      	testq	%r13, %r13
 4fd85bf:      	js	0x4fd86df <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x25f>
 4fd85c5:      	leaq	-0x27ef8d4(%rip), %rax  # 0x27e8cf8 <_RNvNtNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7flt2dec8strategy6dragon9POW5TO256+0x2ac>
 4fd85cc:      	mulss	(%rax,%r13,4), %xmm0
 4fd85d2:      	jmp	0x4fd86f1 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x271>
 4fd85d7:      	movzbl	(%r14), %edx
 4fd85db:      	movzwl	0x1(%r14), %eax
 4fd85e0:      	shll	$0x8, %eax
 4fd85e3:      	orl	%edx, %eax
 4fd85e5:      	andl	$0xdfdfdf, %eax         # imm = 0xDFDFDF
 4fd85ea:      	cmpl	$0x464e49, %eax         # imm = 0x464E49
 4fd85ef:      	je	0x4fd860c <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x18c>
 4fd85f1:      	cmpl	$0x4e414e, %eax         # imm = 0x4E414E
 4fd85f6:      	jne	0x4fd8602 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x182>
 4fd85f8:      	movss	-0x4c473bc(%rip), %xmm0 # 0x391244 <anon.cb993e553e62fd61af54f61042394abd.37.llvm.13524013805524886833+0xc>
 4fd8600:      	jmp	0x4fd8614 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x194>
 4fd8602:      	movl	$0x100, %edx            # imm = 0x100
 4fd8607:      	jmp	0x4fd8705 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x285>
 4fd860c:      	movss	-0x4c493dc(%rip), %xmm0 # 0x38f238 <anon.1b0e954ecefa2908054dc779db4cbc3d.723.llvm.10163399475428358617+0x40>
 4fd8614:      	cmpb	$0x2d, %bl
 4fd8617:      	jne	0x4fd8620 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x1a0>
 4fd8619:      	xorps	-0x4c511a0(%rip), %xmm0 # 0x387480 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x160>
 4fd8620:      	xorl	%ecx, %ecx
 4fd8622:      	movl	$0x100, %edx            # imm = 0x100
 4fd8627:      	jmp	0x4fd8705 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x285>
 4fd862c:      	movq	%rax, %r15
 4fd862f:      	testl	%edx, %edx
 4fd8631:      	movl	-0x2c(%rbp), %r13d
 4fd8635:      	jns	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd8637:      	jmp	0x4fd8670 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x1f0>
 4fd8639:      	leaq	-0x27f55f0(%rip), %rcx  # 0x27e3050 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x84>
 4fd8640:      	movq	%rbx, %rax
 4fd8643:      	mulq	-0x50(%rcx,%r13,8)
 4fd8648:      	seto	%cl
 4fd864b:      	notb	%cl
 4fd864d:      	cmpq	$0x1000001, %rax        # imm = 0x1000001
 4fd8653:      	setb	%dl
 4fd8656:      	testb	%dl, %cl
 4fd8658:      	jne	0x4fd86d0 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x250>
 4fd865a:      	movq	%r13, %rdi
 4fd865d:      	movq	%rbx, %rsi
 4fd8660:      	callq	0x4fcc230 <_RINvNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7dec2flt6lemire13compute_floatfEBa_>
 4fd8665:      	movq	%rax, %r15
 4fd8668:      	testl	%edx, %edx
 4fd866a:      	movl	-0x2c(%rbp), %r13d
 4fd866e:      	jns	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd8670:      	leaq	-0x340(%rbp), %rdi
 4fd8677:      	movq	%r14, %rsi
 4fd867a:      	movq	%r12, %rdx
 4fd867d:      	callq	*0x173f0d(%rip)         # 0x514c590 <writev+0x514c590>
 4fd8683:      	cmpq	$0x0, -0x340(%rbp)
 4fd868b:      	sete	%cl
 4fd868e:      	movl	-0x38(%rbp), %eax
 4fd8691:      	cmpl	$0xfffffebc, %eax       # imm = 0xFFFFFEBC
 4fd8696:      	setl	%dl
 4fd8699:      	orb	%cl, %dl
 4fd869b:      	je	0x4fd86a1 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x221>
 4fd869d:      	xorl	%edx, %edx
 4fd869f:      	jmp	0x4fd86b0 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x230>
 4fd86a1:      	movl	$0xff, %edx
 4fd86a6:      	xorl	%r14d, %r14d
 4fd86a9:      	cmpl	$0x135, %eax            # imm = 0x135
 4fd86ae:      	jle	0x4fd8725 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x2a5>
 4fd86b0:      	xorl	%r15d, %r15d
 4fd86b3:      	shll	$0x17, %edx
 4fd86b6:      	orl	%edx, %r15d
 4fd86b9:      	movd	%r15d, %xmm0
 4fd86be:      	cmpb	$0x2d, %r13b
 4fd86c2:      	jne	0x4fd86cc <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x24c>
 4fd86c4:      	pxor	-0x4c5124c(%rip), %xmm0 # 0x387480 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x160>
 4fd86cc:      	xorl	%edx, %edx
 4fd86ce:      	jmp	0x4fd8703 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x283>
 4fd86d0:      	cvtsi2ss	%rax, %xmm0
 4fd86d5:      	mulss	-0x4c4a5c9(%rip), %xmm0 # 0x38e114 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.2445.llvm.6417388326552730234+0x24>
 4fd86dd:      	jmp	0x4fd86f1 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x271>
 4fd86df:      	shlq	$0x2, %r13
 4fd86e3:      	leaq	-0x27ef9f2(%rip), %rax  # 0x27e8cf8 <_RNvNtNtNtNtNtCs4NRVxsYgnAr_4core3num3imp7flt2dec8strategy6dragon9POW5TO256+0x2ac>
 4fd86ea:      	subq	%r13, %rax
 4fd86ed:      	divss	(%rax), %xmm0
 4fd86f1:      	cmpb	$0x2d, -0x2c(%rbp)
 4fd86f5:      	jne	0x4fd86fe <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x27e>
 4fd86f7:      	xorps	-0x4c5127e(%rip), %xmm0 # 0x387480 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.9420966449765720106+0x160>
 4fd86fe:      	movl	$0x100, %edx            # imm = 0x100
 4fd8703:      	xorl	%ecx, %ecx
 4fd8705:      	addq	$0x318, %rsp            # imm = 0x318
 4fd870c:      	popq	%rbx
 4fd870d:      	popq	%r12
 4fd870f:      	popq	%r13
 4fd8711:      	popq	%r14
 4fd8713:      	popq	%r15
 4fd8715:      	popq	%rbp
 4fd8716:      	movd	%xmm0, %eax
 4fd871a:      	shlq	$0x20, %rax
 4fd871e:      	orq	%rdx, %rax
 4fd8721:      	orq	%rcx, %rax
 4fd8724:      	retq
 4fd8725:      	testl	%eax, %eax
 4fd8727:      	jle	0x4fd8771 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x2f1>
 4fd8729:      	xorl	%r14d, %r14d
 4fd872c:      	leaq	-0x340(%rbp), %rbx
 4fd8733:      	movq	0x173dde(%rip), %r12    # 0x514c518 <writev+0x514c518>
 4fd873a:      	movl	$0x3c, %r15d
 4fd8740:      	cmpl	$0x13, %eax
 4fd8743:      	jae	0x4fd8753 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x2d3>
 4fd8745:      	movl	%eax, %eax
 4fd8747:      	leaq	-0x27f54aa(%rip), %rcx  # 0x27e32a4 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2d8>
 4fd874e:      	movzbl	(%rax,%rcx), %r15d
 4fd8753:      	movq	%rbx, %rdi
 4fd8756:      	movq	%r15, %rsi
 4fd8759:      	callq	*%r12
 4fd875c:      	movl	-0x38(%rbp), %eax
 4fd875f:      	cmpl	$0xfffff800, %eax       # imm = 0xFFFFF800
 4fd8764:      	jle	0x4fd869d <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x21d>
 4fd876a:      	addl	%r15d, %r14d
 4fd876d:      	testl	%eax, %eax
 4fd876f:      	jg	0x4fd873a <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x2ba>
 4fd8771:      	xorl	%r15d, %r15d
 4fd8774:      	leaq	-0x340(%rbp), %rbx
 4fd877b:      	movq	0x173d8e(%rip), %r13    # 0x514c510 <writev+0x514c510>
 4fd8782:      	testl	%eax, %eax
 4fd8784:      	je	0x4fd87a3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x323>
 4fd8786:      	negl	%eax
 4fd8788:      	movl	$0x3c, %r12d
 4fd878e:      	cmpl	$0x13, %eax
 4fd8791:      	jae	0x4fd87ba <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x33a>
 4fd8793:      	movl	%eax, %eax
 4fd8795:      	leaq	-0x27f54f8(%rip), %rcx  # 0x27e32a4 <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x2d8>
 4fd879c:      	movzbl	(%rax,%rcx), %r12d
 4fd87a1:      	jmp	0x4fd87ba <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x33a>
 4fd87a3:      	movzbl	-0x338(%rbp), %eax
 4fd87aa:      	cmpb	$0x4, %al
 4fd87ac:      	ja	0x4fd87d4 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x354>
 4fd87ae:      	cmpb	$0x2, %al
 4fd87b0:      	movl	$0x0, %r12d
 4fd87b6:      	adcq	$0x1, %r12
 4fd87ba:      	movq	%rbx, %rdi
 4fd87bd:      	movq	%r12, %rsi
 4fd87c0:      	callq	*%r13
 4fd87c3:      	movl	-0x38(%rbp), %eax
 4fd87c6:      	cmpl	$0x7ff, %eax            # imm = 0x7FF
 4fd87cb:      	jg	0x4fd8825 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x3a5>
 4fd87cd:      	subl	%r12d, %r14d
 4fd87d0:      	testl	%eax, %eax
 4fd87d2:      	jle	0x4fd8782 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x302>
 4fd87d4:      	decl	%r14d
 4fd87d7:      	cmpl	$-0x7f, %r14d
 4fd87db:      	jg	0x4fd8817 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x397>
 4fd87dd:      	movl	$0x3c, %r15d
 4fd87e3:      	leaq	-0x340(%rbp), %rbx
 4fd87ea:      	movq	0x173d27(%rip), %r12    # 0x514c518 <writev+0x514c518>
 4fd87f1:      	movl	%r14d, %r13d
 4fd87f4:      	movl	$0xffffff82, %r14d      # imm = 0xFFFFFF82
 4fd87fa:      	subl	%r13d, %r14d
 4fd87fd:      	cmpl	$0x3c, %r14d
 4fd8801:      	cmovael	%r15d, %r14d
 4fd8805:      	movq	%rbx, %rdi
 4fd8808:      	movq	%r14, %rsi
 4fd880b:      	callq	*%r12
 4fd880e:      	addl	%r13d, %r14d
 4fd8811:      	cmpl	$-0x7e, %r14d
 4fd8815:      	jb	0x4fd87f1 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x371>
 4fd8817:      	leal	0x7f(%r14), %eax
 4fd881b:      	cmpl	$0xfe, %eax
 4fd8820:      	jle	0x4fd8833 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x3b3>
 4fd8822:      	xorl	%r15d, %r15d
 4fd8825:      	movl	-0x2c(%rbp), %r13d
 4fd8829:      	movl	$0xff, %edx
 4fd882e:      	jmp	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd8833:      	leaq	-0x340(%rbp), %rdi
 4fd883a:      	movl	$0x18, %esi
 4fd883f:      	callq	*0x173ccb(%rip)         # 0x514c510 <writev+0x514c510>
 4fd8845:      	movq	-0x340(%rbp), %rcx
 4fd884c:      	testq	%rcx, %rcx
 4fd884f:      	movl	-0x2c(%rbp), %r13d
 4fd8853:      	je	0x4fd8877 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x3f7>
 4fd8855:      	movslq	-0x38(%rbp), %rax
 4fd8859:      	testq	%rax, %rax
 4fd885c:      	js	0x4fd8877 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x3f7>
 4fd885e:      	cmpl	$0x12, %eax
 4fd8861:      	ja	0x4fd8941 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4c1>
 4fd8867:      	testl	%eax, %eax
 4fd8869:      	je	0x4fd8886 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x406>
 4fd886b:      	cmpl	$0x1, %eax
 4fd886e:      	jne	0x4fd888b <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x40b>
 4fd8870:      	xorl	%r15d, %r15d
 4fd8873:      	xorl	%edx, %edx
 4fd8875:      	jmp	0x4fd88e0 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x460>
 4fd8877:      	addl	$0x7e, %r14d
 4fd887b:      	xorl	%r15d, %r15d
 4fd887e:      	movl	%r14d, %edx
 4fd8881:      	jmp	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd8886:      	xorl	%r15d, %r15d
 4fd8889:      	jmp	0x4fd88f7 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x477>
 4fd888b:      	movl	%eax, %esi
 4fd888d:      	andl	$0x1e, %esi
 4fd8890:      	xorl	%r15d, %r15d
 4fd8893:      	xorl	%edi, %edi
 4fd8895:      	jmp	0x4fd889f <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x41f>
 4fd8897:      	incq	%rdi
 4fd889a:      	cmpq	%rsi, %rdi
 4fd889d:      	je	0x4fd88d8 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x458>
 4fd889f:      	movq	%rdi, %rdx
 4fd88a2:      	addq	%r15, %r15
 4fd88a5:      	leaq	(%r15,%r15,4), %rdi
 4fd88a9:      	cmpq	%rcx, %rdx
 4fd88ac:      	jae	0x4fd88ba <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x43a>
 4fd88ae:      	movzbl	-0x338(%rbp,%rdx), %r8d
 4fd88b7:      	addq	%r8, %rdi
 4fd88ba:      	addq	%rdi, %rdi
 4fd88bd:      	leaq	(%rdi,%rdi,4), %r15
 4fd88c1:      	leaq	0x1(%rdx), %rdi
 4fd88c5:      	cmpq	%rcx, %rdi
 4fd88c8:      	jae	0x4fd8897 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x417>
 4fd88ca:      	movzbl	-0x337(%rbp,%rdx), %r8d
 4fd88d3:      	addq	%r8, %r15
 4fd88d6:      	jmp	0x4fd8897 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x417>
 4fd88d8:      	testb	$0x1, %al
 4fd88da:      	je	0x4fd88f7 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x477>
 4fd88dc:      	addq	$0x2, %rdx
 4fd88e0:      	addq	%r15, %r15
 4fd88e3:      	leaq	(%r15,%r15,4), %r15
 4fd88e7:      	cmpq	%rcx, %rdx
 4fd88ea:      	jae	0x4fd88f7 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x477>
 4fd88ec:      	movzbl	-0x338(%rbp,%rdx), %edx
 4fd88f4:      	addq	%rdx, %r15
 4fd88f7:      	cmpq	%rax, %rcx
 4fd88fa:      	jbe	0x4fd8938 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b8>
 4fd88fc:      	movzbl	-0x338(%rbp,%rax), %edx
 4fd8904:      	cmpb	$0x5, %dl
 4fd8907:      	sete	%sil
 4fd890b:      	leaq	0x1(%rax), %rdi
 4fd890f:      	cmpq	%rcx, %rdi
 4fd8912:      	sete	%cl
 4fd8915:      	testb	%sil, %cl
 4fd8918:      	je	0x4fd8930 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b0>
 4fd891a:      	cmpb	$0x0, -0x34(%rbp)
 4fd891e:      	jne	0x4fd8935 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b5>
 4fd8920:      	testl	%eax, %eax
 4fd8922:      	je	0x4fd8938 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b8>
 4fd8924:      	testb	$0x1, -0x339(%rbp,%rax)
 4fd892c:      	jne	0x4fd8935 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b5>
 4fd892e:      	jmp	0x4fd8938 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b8>
 4fd8930:      	cmpb	$0x4, %dl
 4fd8933:      	jbe	0x4fd8938 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4b8>
 4fd8935:      	incq	%r15
 4fd8938:      	cmpq	$0x1000000, %r15        # imm = 0x1000000
 4fd893f:      	jb	0x4fd8980 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x500>
 4fd8941:      	leaq	-0x340(%rbp), %rbx
 4fd8948:      	movl	$0x1, %esi
 4fd894d:      	movq	%rbx, %rdi
 4fd8950:      	callq	*0x173bc2(%rip)         # 0x514c518 <writev+0x514c518>
 4fd8956:      	movq	%rbx, %rdi
 4fd8959:      	callq	*0x173bc1(%rip)         # 0x514c520 <writev+0x514c520>
 4fd895f:      	movq	%rax, %r15
 4fd8962:      	leal	0x80(%r14), %eax
 4fd8969:      	cmpl	$0xfe, %eax
 4fd896e:      	jle	0x4fd897d <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x4fd>
 4fd8970:      	xorl	%r15d, %r15d
 4fd8973:      	movl	$0xff, %edx
 4fd8978:      	jmp	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd897d:      	incl	%r14d
 4fd8980:      	cmpq	$0x800000, %r15         # imm = 0x800000
 4fd8987:      	movl	$0x7f, %edx
 4fd898c:      	sbbl	$0x0, %edx
 4fd898f:      	addl	%r14d, %edx
 4fd8992:      	andl	$0x7fffff, %r15d        # imm = 0x7FFFFF
 4fd8999:      	jmp	0x4fd86b3 <_RNvXs1_NtNtCs4NRVxsYgnAr_4core3num11float_parsefNtNtNtB9_3str6traits7FromStr8from_str+0x233>
 4fd899e:      	int3
 4fd899f:      	int3

0000000004fda0b0 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next>:
 4fda0b0:      	movq	%rdi, %rax
 4fda0b3:      	movq	0x8(%rsi), %rdx
 4fda0b7:      	testq	%rdx, %rdx
 4fda0ba:      	je	0x4fda235 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x185>
 4fda0c0:      	pushq	%r14
 4fda0c2:      	pushq	%rbx
 4fda0c3:      	movq	(%rsi), %rcx
 4fda0c6:      	xorl	%edi, %edi
 4fda0c8:      	leaq	-0x27f6862(%rip), %r9   # 0x27e386d <_RNvNtNtNtCs4NRVxsYgnAr_4core7unicode12unicode_data9uppercase17BITSET_CHUNKS_MAP+0x8a1>
 4fda0cf:      	leaq	-0x4c33d42(%rip), %r10  # 0x3a6394 <anon.f7b1e62628d2017b05436ef631806f29.78.llvm.12854340704546893908>
 4fda0d6:      	jmp	0x4fda0eb <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x3b>
 4fda0d8:      	addq	$0x4, %rdi
 4fda0dc:      	movq	%rdi, %r8
 4fda0df:      	movq	%r8, %rdi
 4fda0e2:      	cmpq	%rdx, %r8
 4fda0e5:      	jae	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda0eb:      	movzbl	(%rcx,%rdi), %r11d
 4fda0f0:      	leaq	0x1(%rdi), %r8
 4fda0f4:      	testb	%r11b, %r11b
 4fda0f7:      	jns	0x4fda0df <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x2f>
 4fda0f9:      	movzbl	(%r11,%r9), %ebx
 4fda0fe:      	cmpl	$0x4, %ebx
 4fda101:      	je	0x4fda156 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0xa6>
 4fda103:      	cmpl	$0x3, %ebx
 4fda106:      	je	0x4fda12c <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x7c>
 4fda108:      	cmpl	$0x2, %ebx
 4fda10b:      	jne	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda111:      	leaq	(%rcx,%r8), %r11
 4fda115:      	cmpq	%rdx, %r8
 4fda118:      	cmovaeq	%r10, %r11
 4fda11c:      	cmpb	$-0x40, (%r11)
 4fda120:      	jge	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda126:      	addq	$0x2, %rdi
 4fda12a:      	jmp	0x4fda0dc <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x2c>
 4fda12c:      	leaq	(%rcx,%r8), %rbx
 4fda130:      	cmpq	%rdx, %r8
 4fda133:      	cmovaeq	%r10, %rbx
 4fda137:      	movzbl	(%rbx), %ebx
 4fda13a:      	cmpq	$0xe0, %r11
 4fda141:      	je	0x4fda180 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0xd0>
 4fda143:      	cmpl	$0xed, %r11d
 4fda14a:      	jne	0x4fda197 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0xe7>
 4fda14c:      	cmpb	$-0x61, %bl
 4fda14f:      	jle	0x4fda1b0 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x100>
 4fda151:      	jmp	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda156:      	leaq	(%rcx,%r8), %rbx
 4fda15a:      	cmpq	%rdx, %r8
 4fda15d:      	cmovaeq	%r10, %rbx
 4fda161:      	movzbl	(%rbx), %ebx
 4fda164:      	cmpq	$0xf0, %r11
 4fda16b:      	je	0x4fda18d <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0xdd>
 4fda16d:      	cmpl	$0xf4, %r11d
 4fda174:      	jne	0x4fda1cf <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x11f>
 4fda176:      	cmpb	$-0x71, %bl
 4fda179:      	jle	0x4fda1de <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x12e>
 4fda17b:      	jmp	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda180:      	andb	$-0x20, %bl
 4fda183:      	cmpb	$-0x60, %bl
 4fda186:      	je	0x4fda1b0 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x100>
 4fda188:      	jmp	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda18d:      	addb	$0x70, %bl
 4fda190:      	cmpb	$0x30, %bl
 4fda193:      	jb	0x4fda1de <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x12e>
 4fda195:      	jmp	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda197:      	leal	0x1f(%r11), %r14d
 4fda19b:      	cmpb	$0xc, %r14b
 4fda19f:      	jb	0x4fda1ab <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0xfb>
 4fda1a1:      	andb	$-0x2, %r11b
 4fda1a5:      	cmpb	$-0x12, %r11b
 4fda1a9:      	jne	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda1ab:      	cmpb	$-0x40, %bl
 4fda1ae:      	jge	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda1b0:      	leaq	0x2(%rdi), %r8
 4fda1b4:      	cmpq	%rdx, %r8
 4fda1b7:      	leaq	0x2(%rcx,%rdi), %r11
 4fda1bc:      	cmovaeq	%r10, %r11
 4fda1c0:      	cmpb	$-0x40, (%r11)
 4fda1c4:      	jge	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda1c6:      	addq	$0x3, %rdi
 4fda1ca:      	jmp	0x4fda0dc <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x2c>
 4fda1cf:      	addb	$0xf, %r11b
 4fda1d3:      	cmpb	$0x2, %r11b
 4fda1d7:      	ja	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda1d9:      	cmpb	$-0x40, %bl
 4fda1dc:      	jge	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda1de:      	leaq	0x2(%rdi), %r8
 4fda1e2:      	cmpq	%rdx, %r8
 4fda1e5:      	leaq	0x2(%rcx,%rdi), %r11
 4fda1ea:      	cmovaeq	%r10, %r11
 4fda1ee:      	cmpb	$-0x41, (%r11)
 4fda1f2:      	jg	0x4fda20e <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x15e>
 4fda1f4:      	leaq	0x3(%rdi), %r8
 4fda1f8:      	cmpq	%rdx, %r8
 4fda1fb:      	leaq	0x3(%rcx,%rdi), %r11
 4fda200:      	cmovaeq	%r10, %r11
 4fda204:      	cmpb	$-0x41, (%r11)
 4fda208:      	jle	0x4fda0d8 <_RNvXs2_NtNtCs4NRVxsYgnAr_4core3str5lossyNtB5_10Utf8ChunksNtNtNtNtB9_4iter6traits8iterator8Iterator4next+0x28>
 4fda20e:      	leaq	(%rcx,%r8), %r9
 4fda212:      	subq	%r8, %rdx
 4fda215:      	movq	%r9, (%rsi)
 4fda218:      	movq	%rdx, 0x8(%rsi)
 4fda21c:      	subq	%rdi, %r8
 4fda21f:      	movq	%rcx, (%rax)
 4fda222:      	addq	%rdi, %rcx
 4fda225:      	movq	%rdi, 0x8(%rax)
 4fda229:      	movq	%rcx, 0x10(%rax)
 4fda22d:      	movq	%r8, 0x18(%rax)
 4fda231:      	popq	%rbx
 4fda232:      	popq	%r14
 4fda234:      	retq
 4fda235:      	movq	$0x0, (%rax)
 4fda23c:      	retq
 4fda23d:      	int3
 4fda23e:      	int3
 4fda23f:      	int3

0000000004fdb310 <_RNvXs8_NtCs4NRVxsYgnAr_4core3fmtNtB5_9ArgumentsNtB5_7Display3fmt>:
 4fdb310:      	movq	(%rsi), %rax
 4fdb313:      	movq	0x8(%rsi), %rsi
 4fdb317:      	movq	(%rdi), %rdx
 4fdb31a:      	movq	0x8(%rdi), %rcx
 4fdb31e:      	movq	%rax, %rdi
 4fdb321:      	jmpq	*0x166631(%rip)         # 0x5141958 <writev+0x5141958>
 4fdb327:      	int3
 4fdb328:      	int3
 4fdb329:      	int3
 4fdb32a:      	int3
 4fdb32b:      	int3
 4fdb32c:      	int3
 4fdb32d:      	int3
 4fdb32e:      	int3
 4fdb32f:      	int3

0000000004fdc3d0 <_RNvXsi_NtNtNtCs4NRVxsYgnAr_4core3fmt3num3impjNtB9_7Display3fmt>:
 4fdc3d0:      	pushq	%rbp
 4fdc3d1:      	movq	%rsp, %rbp
 4fdc3d4:      	pushq	%r14
 4fdc3d6:      	pushq	%rbx
 4fdc3d7:      	subq	$0x20, %rsp
 4fdc3db:      	movq	%rsi, %rbx
 4fdc3de:      	movq	(%rdi), %rdi
 4fdc3e1:      	leaq	-0x24(%rbp), %rsi
 4fdc3e5:      	movl	$0x14, %r14d
 4fdc3eb:      	movl	$0x14, %edx
 4fdc3f0:      	callq	0x4fd0b80 <_RNvMsf_NtNtNtCs4NRVxsYgnAr_4core3fmt3num3impy10__fmt_inner>
 4fdc3f5:      	subq	%rax, %r14
 4fdc3f8:      	leaq	(%rax,%rbp), %r8
 4fdc3fc:      	addq	$-0x24, %r8
 4fdc400:      	movl	$0x1, %edx
 4fdc405:      	movq	%rbx, %rdi
 4fdc408:      	movl	$0x1, %esi
 4fdc40d:      	xorl	%ecx, %ecx
 4fdc40f:      	movq	%r14, %r9
 4fdc412:      	callq	*0x170110(%rip)         # 0x514c528 <writev+0x514c528>
 4fdc418:      	addq	$0x20, %rsp
 4fdc41c:      	popq	%rbx
 4fdc41d:      	popq	%r14
 4fdc41f:      	popq	%rbp
 4fdc420:      	retq
 4fdc421:      	int3
 4fdc422:      	int3
 4fdc423:      	int3
 4fdc424:      	int3
 4fdc425:      	int3
 4fdc426:      	int3
 4fdc427:      	int3
 4fdc428:      	int3
 4fdc429:      	int3
 4fdc42a:      	int3
 4fdc42b:      	int3
 4fdc42c:      	int3
 4fdc42d:      	int3
 4fdc42e:      	int3
 4fdc42f:      	int3

0000000004fde5e0 <ceil>:
 4fde5e0:      	movq	%xmm0, %rax
 4fde5e5:      	movq	%rax, %rcx
 4fde5e8:      	shrq	$0x34, %rcx
 4fde5ec:      	andl	$0x7ff, %ecx            # imm = 0x7FF
 4fde5f2:      	cmpl	$0x432, %ecx            # imm = 0x432
 4fde5f8:      	ja	0x4fde63d <ceil+0x5d>
 4fde5fa:      	cmpl	$0x3fe, %ecx            # imm = 0x3FE
 4fde600:      	jbe	0x4fde63e <ceil+0x5e>
 4fde602:      	addl	$0xfffffc01, %ecx       # imm = 0xFFFFFC01
 4fde608:      	movabsq	$0xfffffffffffff, %rdx  # imm = 0xFFFFFFFFFFFFF
 4fde612:      	shrq	%cl, %rdx
 4fde615:      	testq	%rax, %rdx
 4fde618:      	je	0x4fde63d <ceil+0x5d>
 4fde61a:      	movl	%ecx, %ecx
 4fde61c:      	xorl	%esi, %esi
 4fde61e:      	testq	%rax, %rax
 4fde621:      	cmovsq	%rsi, %rdx
 4fde625:      	movabsq	$-0x10000000000000, %rsi # imm = 0xFFF0000000000000
 4fde62f:      	sarq	%cl, %rsi
 4fde632:      	addq	%rax, %rdx
 4fde635:      	andq	%rdx, %rsi
 4fde638:      	movq	%rsi, %xmm0
 4fde63d:      	retq
 4fde63e:      	testq	%rax, %rax
 4fde641:      	js	0x4fde64e <ceil+0x6e>
 4fde643:      	je	0x4fde63d <ceil+0x5d>
 4fde645:      	movsd	-0x4c54415(%rip), %xmm0 # 0x38a238 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.633.llvm.6417388326552730234+0x30>
 4fde64d:      	retq
 4fde64e:      	movsd	-0x4c53dd6(%rip), %xmm0 # 0x38a880 <anon.cb993e553e62fd61af54f61042394abd.23.llvm.13524013805524886833+0x178>
 4fde656:      	retq
 4fde657:      	int3
 4fde658:      	int3
 4fde659:      	int3
 4fde65a:      	int3
 4fde65b:      	int3
 4fde65c:      	int3
 4fde65d:      	int3
 4fde65e:      	int3
 4fde65f:      	int3
