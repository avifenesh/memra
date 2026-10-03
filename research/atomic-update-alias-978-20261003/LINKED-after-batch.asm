
/home/avifenesh/.local/state/memra-rig-darklanes-20261002/receipts/B/atomic-update-978/emit-after-server/server-tests-after:	file format elf64-x86-64

Disassembly of section .text:

000000000352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>:
 352d6c0:      	cmpl	$-0x1, (%rdi)
 352d6c3:      	jne	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
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

000000000352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>:
 352d8b0:      	pushq	%r15
 352d8b2:      	pushq	%r14
 352d8b4:      	pushq	%rbx
 352d8b5:      	movq	%rdi, %rbx
 352d8b8:      	callq	0x3550c90 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs1S6izGDSMYD_4http6header3map9HeaderMapECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 352d8bd:      	movq	0x60(%rbx), %r15
 352d8c1:      	testq	%r15, %r15
 352d8c4:      	je	0x352d8e1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x31>
 352d8c6:      	movq	%r15, %rdi
 352d8c9:      	callq	0x392c3b0 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCs4NRVxsYgnAr_4core3any6TypeIdINtNtCscdodAO9FK5_5alloc5boxed3BoxDNtNtCs1S6izGDSMYD_4http10extensions8AnyCloneNtNtBT_6marker4SendNtB2G_4SyncEL_EEENtNtNtBT_3ops4drop4Drop4dropCs3pwlnhBXFtN_12memra_server>
 352d8ce:      	movl	$0x20, %esi
 352d8d3:      	movl	$0x8, %edx
 352d8d8:      	movq	%r15, %rdi
 352d8db:      	callq	*0x1c1345f(%rip)        # 0x5140d40 <writev+0x5140d40>
 352d8e1:      	movq	0x70(%rbx), %r15
 352d8e5:      	movq	0x78(%rbx), %rbx
 352d8e9:      	movq	(%rbx), %rax
 352d8ec:      	testq	%rax, %rax
 352d8ef:      	je	0x352d8f6 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x46>
 352d8f1:      	movq	%r15, %rdi
 352d8f4:      	callq	*%rax
 352d8f6:      	movq	0x8(%rbx), %rsi
 352d8fa:      	testq	%rsi, %rsi
 352d8fd:      	je	0x352d911 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x61>
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
 352d921:      	je	0x352d969 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0xb9>
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
 352d94e:      	jmp	0x352d95c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0xac>
 352d950:      	movq	%rax, %r14
 352d953:      	movq	0x60(%rbx), %rdi
 352d957:      	callq	0x354d360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs1S6izGDSMYD_4http10extensions10ExtensionsECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 352d95c:      	movq	0x70(%rbx), %rdi
 352d960:      	movq	0x78(%rbx), %rsi
 352d964:      	callq	0x352fd30 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCsaSG9NyffgI5_9axum_core4body4BodyECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
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
 354c165:      	jmp	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
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
 354c19a:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 354c19f:      	movq	%r14, %rdi
 354c1a2:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 354c1a7:      	callq	*0x1bf4c0b(%rip)        # 0x5140db8 <writev+0x5140db8>
 354c1ad:      	int3
 354c1ae:      	int3
 354c1af:      	int3

000000000354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649>:
 354c4e0:      	pushq	%r14
 354c4e2:      	pushq	%rbx
 354c4e3:      	pushq	%rax
 354c4e4:      	movq	%rdi, %rbx
 354c4e7:      	callq	0x3c723b0 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 354c4ec:      	movq	0x28(%rbx), %rax
 354c4f0:      	lock
 354c4f1:      	decq	(%rax)
 354c4f4:      	jne	0x354c4ff <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x1f>
 354c4f6:      	leaq	0x28(%rbx), %rdi
 354c4fa:      	callq	0x39ba050 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtB7_3vec3VecNtNtB7_6string6StringEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c4ff:      	movq	0x30(%rbx), %rax
 354c503:      	lock
 354c504:      	decq	(%rax)
 354c507:      	jne	0x354c512 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x32>
 354c509:      	leaq	0x30(%rbx), %rdi
 354c50d:      	callq	0x39bb0f0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEE9drop_slowB1Z_>
 354c512:      	movq	0x38(%rbx), %rax
 354c516:      	lock
 354c517:      	decq	(%rax)
 354c51a:      	jne	0x354c525 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x45>
 354c51c:      	leaq	0x38(%rbx), %rdi
 354c520:      	callq	0x39bc0a0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockIBw_NtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEE9drop_slowB1C_>
 354c525:      	movq	0x80(%rbx), %rax
 354c52c:      	testq	%rax, %rax
 354c52f:      	je	0x354c543 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x63>
 354c531:      	lock
 354c532:      	decq	(%rax)
 354c535:      	jne	0x354c543 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x63>
 354c537:      	leaq	0x80(%rbx), %rdi
 354c53e:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c543:      	movq	0x90(%rbx), %rax
 354c54a:      	testq	%rax, %rax
 354c54d:      	je	0x354c561 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x81>
 354c54f:      	lock
 354c550:      	decq	(%rax)
 354c553:      	jne	0x354c561 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x81>
 354c555:      	leaq	0x90(%rbx), %rdi
 354c55c:      	callq	0x39bb080 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringIBw_NtCscIVK9LJ3de3_15memra_tokenizer9TokenizerEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c561:      	movq	0x98(%rbx), %rax
 354c568:      	testq	%rax, %rax
 354c56b:      	je	0x354c580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xa0>
 354c56d:      	lock
 354c56e:      	decq	(%rax)
 354c571:      	jne	0x354c580 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xa0>
 354c573:      	leaq	0x98(%rbx), %rdi
 354c57a:      	callq	*0x1bf5a28(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c580:      	movq	0x10(%rbx), %rax
 354c584:      	testq	%rax, %rax
 354c587:      	je	0x354c599 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xb9>
 354c589:      	lock
 354c58a:      	decq	(%rax)
 354c58d:      	jne	0x354c599 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xb9>
 354c58f:      	leaq	0x10(%rbx), %rdi
 354c593:      	callq	*0x1bf5a0f(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c599:      	movq	0x40(%rbx), %rax
 354c59d:      	lock
 354c59e:      	decq	(%rax)
 354c5a1:      	jne	0x354c5ac <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xcc>
 354c5a3:      	leaq	0x40(%rbx), %rdi
 354c5a7:      	callq	0x39bb8d0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEE9drop_slowB1y_>
 354c5ac:      	movq	0x48(%rbx), %rax
 354c5b0:      	lock
 354c5b1:      	decq	(%rax)
 354c5b4:      	jne	0x354c5bf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xdf>
 354c5b6:      	leaq	0x48(%rbx), %rdi
 354c5ba:      	callq	0x39b9f90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6worker15EventQueueStateE9drop_slowBJ_>
 354c5bf:      	movq	0x50(%rbx), %rax
 354c5c3:      	lock
 354c5c4:      	decq	(%rax)
 354c5c7:      	jne	0x354c5d2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0xf2>
 354c5c9:      	leaq	0x50(%rbx), %rdi
 354c5cd:      	callq	0x39bb500 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringjEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c5d2:      	movq	0x58(%rbx), %rax
 354c5d6:      	lock
 354c5d7:      	decq	(%rax)
 354c5da:      	jne	0x354c5e5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x105>
 354c5dc:      	leaq	0x58(%rbx), %rdi
 354c5e0:      	callq	0x39bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 354c5e5:      	movq	0xb0(%rbx), %rax
 354c5ec:      	testq	%rax, %rax
 354c5ef:      	je	0x354c603 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x123>
 354c5f1:      	lock
 354c5f2:      	decq	(%rax)
 354c5f5:      	jne	0x354c603 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x123>
 354c5f7:      	leaq	0xb0(%rbx), %rdi
 354c5fe:      	callq	0x39bd090 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server8darklane10BgJobStateE9drop_slowBJ_>
 354c603:      	movq	0x60(%rbx), %rax
 354c607:      	lock
 354c608:      	decq	(%rax)
 354c60b:      	jne	0x354c616 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x136>
 354c60d:      	leaq	0x60(%rbx), %rdi
 354c611:      	callq	0x39bbe90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCsdMwdNqnNPrU_11memra_lanes12audio_stream14AudioSchedulerEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c616:      	movq	0x68(%rbx), %rax
 354c61a:      	lock
 354c61b:      	decq	(%rax)
 354c61e:      	jne	0x354c629 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x149>
 354c620:      	leaq	0x68(%rbx), %rdi
 354c624:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c629:      	movq	0x78(%rbx), %rax
 354c62d:      	lock
 354c62e:      	decq	(%rax)
 354c631:      	jne	0x354c646 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x166>
 354c633:      	addq	$0x78, %rbx
 354c637:      	movq	%rbx, %rdi
 354c63a:      	addq	$0x8, %rsp
 354c63e:      	popq	%rbx
 354c63f:      	popq	%r14
 354c641:      	jmp	0x39bb490 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server15background_jobs7ControlEEE9drop_slowB2z_>
 354c646:      	addq	$0x8, %rsp
 354c64a:      	popq	%rbx
 354c64b:      	popq	%r14
 354c64d:      	retq
 354c64e:      	movq	%rax, %r14
 354c651:      	jmp	0x354c7cf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2ef>
 354c656:      	movq	%rax, %r14
 354c659:      	jmp	0x354c765 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x285>
 354c65e:      	movq	%rax, %r14
 354c661:      	jmp	0x354c74c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x26c>
 354c666:      	movq	%rax, %r14
 354c669:      	jmp	0x354c72d <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x24d>
 354c66e:      	movq	%rax, %r14
 354c671:      	jmp	0x354c70f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x22f>
 354c676:      	movq	%rax, %r14
 354c679:      	jmp	0x354c7f5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x315>
 354c67e:      	movq	%rax, %r14
 354c681:      	jmp	0x354c7e2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x302>
 354c686:      	movq	%rax, %r14
 354c689:      	jmp	0x354c7b1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2d1>
 354c68e:      	movq	%rax, %r14
 354c691:      	jmp	0x354c79e <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2be>
 354c696:      	movq	%rax, %r14
 354c699:      	jmp	0x354c78b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2ab>
 354c69e:      	movq	%rax, %r14
 354c6a1:      	jmp	0x354c778 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x298>
 354c6a6:      	movq	%rax, %r14
 354c6a9:      	jmp	0x354c6f1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x211>
 354c6ab:      	movq	%rax, %r14
 354c6ae:      	jmp	0x354c6de <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x1fe>
 354c6b0:      	movq	%rax, %r14
 354c6b3:      	jmp	0x354c6cb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x1eb>
 354c6b5:      	movq	%rax, %r14
 354c6b8:      	movq	0x28(%rbx), %rax
 354c6bc:      	lock
 354c6bd:      	decq	(%rax)
 354c6c0:      	jne	0x354c6cb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x1eb>
 354c6c2:      	leaq	0x28(%rbx), %rdi
 354c6c6:      	callq	0x39ba050 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtB7_3vec3VecNtNtB7_6string6StringEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c6cb:      	movq	0x30(%rbx), %rax
 354c6cf:      	lock
 354c6d0:      	decq	(%rax)
 354c6d3:      	jne	0x354c6de <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x1fe>
 354c6d5:      	leaq	0x30(%rbx), %rdi
 354c6d9:      	callq	0x39bb0f0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEE9drop_slowB1Z_>
 354c6de:      	movq	0x38(%rbx), %rax
 354c6e2:      	lock
 354c6e3:      	decq	(%rax)
 354c6e6:      	jne	0x354c6f1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x211>
 354c6e8:      	leaq	0x38(%rbx), %rdi
 354c6ec:      	callq	0x39bc0a0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockIBw_NtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEE9drop_slowB1C_>
 354c6f1:      	movq	0x80(%rbx), %rax
 354c6f8:      	testq	%rax, %rax
 354c6fb:      	je	0x354c70f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x22f>
 354c6fd:      	lock
 354c6fe:      	decq	(%rax)
 354c701:      	jne	0x354c70f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x22f>
 354c703:      	leaq	0x80(%rbx), %rdi
 354c70a:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c70f:      	movq	0x90(%rbx), %rax
 354c716:      	testq	%rax, %rax
 354c719:      	je	0x354c72d <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x24d>
 354c71b:      	lock
 354c71c:      	decq	(%rax)
 354c71f:      	jne	0x354c72d <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x24d>
 354c721:      	leaq	0x90(%rbx), %rdi
 354c728:      	callq	0x39bb080 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringIBw_NtCscIVK9LJ3de3_15memra_tokenizer9TokenizerEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c72d:      	movq	0x98(%rbx), %rax
 354c734:      	testq	%rax, %rax
 354c737:      	je	0x354c74c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x26c>
 354c739:      	lock
 354c73a:      	decq	(%rax)
 354c73d:      	jne	0x354c74c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x26c>
 354c73f:      	leaq	0x98(%rbx), %rdi
 354c746:      	callq	*0x1bf585c(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c74c:      	movq	0x10(%rbx), %rax
 354c750:      	testq	%rax, %rax
 354c753:      	je	0x354c765 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x285>
 354c755:      	lock
 354c756:      	decq	(%rax)
 354c759:      	jne	0x354c765 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x285>
 354c75b:      	leaq	0x10(%rbx), %rdi
 354c75f:      	callq	*0x1bf5843(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 354c765:      	movq	0x40(%rbx), %rax
 354c769:      	lock
 354c76a:      	decq	(%rax)
 354c76d:      	jne	0x354c778 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x298>
 354c76f:      	leaq	0x40(%rbx), %rdi
 354c773:      	callq	0x39bb8d0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEE9drop_slowB1y_>
 354c778:      	movq	0x48(%rbx), %rax
 354c77c:      	lock
 354c77d:      	decq	(%rax)
 354c780:      	jne	0x354c78b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2ab>
 354c782:      	leaq	0x48(%rbx), %rdi
 354c786:      	callq	0x39b9f90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6worker15EventQueueStateE9drop_slowBJ_>
 354c78b:      	movq	0x50(%rbx), %rax
 354c78f:      	lock
 354c790:      	decq	(%rax)
 354c793:      	jne	0x354c79e <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2be>
 354c795:      	leaq	0x50(%rbx), %rdi
 354c799:      	callq	0x39bb500 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringjEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c79e:      	movq	0x58(%rbx), %rax
 354c7a2:      	lock
 354c7a3:      	decq	(%rax)
 354c7a6:      	jne	0x354c7b1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2d1>
 354c7a8:      	leaq	0x58(%rbx), %rdi
 354c7ac:      	callq	0x39bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 354c7b1:      	movq	0xb0(%rbx), %rax
 354c7b8:      	testq	%rax, %rax
 354c7bb:      	je	0x354c7cf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2ef>
 354c7bd:      	lock
 354c7be:      	decq	(%rax)
 354c7c1:      	jne	0x354c7cf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x2ef>
 354c7c3:      	leaq	0xb0(%rbx), %rdi
 354c7ca:      	callq	0x39bd090 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server8darklane10BgJobStateE9drop_slowBJ_>
 354c7cf:      	movq	0x60(%rbx), %rax
 354c7d3:      	lock
 354c7d4:      	decq	(%rax)
 354c7d7:      	jne	0x354c7e2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x302>
 354c7d9:      	leaq	0x60(%rbx), %rdi
 354c7dd:      	callq	0x39bbe90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCsdMwdNqnNPrU_11memra_lanes12audio_stream14AudioSchedulerEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 354c7e2:      	movq	0x68(%rbx), %rax
 354c7e6:      	lock
 354c7e7:      	decq	(%rax)
 354c7ea:      	jne	0x354c7f5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x315>
 354c7ec:      	leaq	0x68(%rbx), %rdi
 354c7f0:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 354c7f5:      	movq	0x78(%rbx), %rax
 354c7f9:      	lock
 354c7fa:      	decq	(%rax)
 354c7fd:      	jne	0x354c80b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649+0x32b>
 354c7ff:      	addq	$0x78, %rbx
 354c803:      	movq	%rbx, %rdi
 354c806:      	callq	0x39bb490 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server15background_jobs7ControlEEE9drop_slowB2z_>
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

0000000003550460 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649>:
 3550460:      	pushq	%rbp
 3550461:      	pushq	%r15
 3550463:      	pushq	%r14
 3550465:      	pushq	%r13
 3550467:      	pushq	%r12
 3550469:      	pushq	%rbx
 355046a:      	pushq	%rax
 355046b:      	movq	0x10(%rdi), %rbx
 355046f:      	testq	%rbx, %rbx
 3550472:      	je	0x355052b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0xcb>
 3550478:      	movq	%rdi, %r14
 355047b:      	movq	0x20(%rdi), %r15
 355047f:      	testq	%r15, %r15
 3550482:      	je	0x35504ee <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0x8e>
 3550484:      	movq	0x8(%r14), %r12
 3550488:      	movdqa	(%r12), %xmm0
 355048e:      	leaq	0x10(%r12), %r13
 3550493:      	pmovmskb	%xmm0, %eax
 3550497:      	notl	%eax
 3550499:      	jmp	0x35504c5 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0x65>
 355049b:      	nopl	(%rax,%rax)
 35504a0:      	leal	-0x1(%rax), %ebp
 35504a3:      	tzcntl	%eax, %ecx
 35504a7:      	andl	%eax, %ebp
 35504a9:      	shll	$0x8, %ecx
 35504ac:      	movq	%r12, %rdi
 35504af:      	subq	%rcx, %rdi
 35504b2:      	addq	$-0x100, %rdi
 35504b9:      	callq	0x38c3440 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryEEB1h_.llvm.9611048873305048036>
 35504be:      	movl	%ebp, %eax
 35504c0:      	decq	%r15
 35504c3:      	je	0x35504ee <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0x8e>
 35504c5:      	testw	%ax, %ax
 35504c8:      	jne	0x35504a0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0x40>
 35504ca:      	nopw	(%rax,%rax)
 35504d0:      	movdqa	(%r13), %xmm0
 35504d6:      	addq	$-0x1000, %r12          # imm = 0xF000
 35504dd:      	addq	$0x10, %r13
 35504e1:      	pmovmskb	%xmm0, %eax
 35504e5:      	xorl	$0xffff, %eax           # imm = 0xFFFF
 35504ea:      	je	0x35504d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0x70>
 35504ec:      	jmp	0x35504a0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0x40>
 35504ee:      	movq	%rbx, %rax
 35504f1:      	shlq	$0x8, %rax
 35504f5:      	addq	%rax, %rbx
 35504f8:      	addq	$0x111, %rbx            # imm = 0x111
 35504ff:      	je	0x355052b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server9job_store16InMemoryJobStoreEBF_.llvm.10122458563676435649+0xcb>
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

0000000003550de0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>:
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
 3550df6:      	jb	0x3550dfd <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x1d>
 3550df8:      	testq	%rcx, %rcx
 3550dfb:      	jne	0x3550e09 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x29>
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
 3550e1c:      	je	0x3550e23 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x43>
 3550e1e:      	movq	%r14, %rdi
 3550e21:      	callq	*%rax
 3550e23:      	movq	0x8(%r12), %rsi
 3550e28:      	testq	%rsi, %rsi
 3550e2b:      	je	0x3550e3b <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x5b>
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
 3550e64:      	je	0x3550e74 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649+0x94>
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
 3552540:      	jmp	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
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

000000000366d170 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649>:
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
 366d255:      	je	0x366dc98 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xb28>
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
 366d2ac:      	callq	0x388f890 <_RNvXs1_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealthNtNtCs4NRVxsYgnAr_4core7default7Default7default>
 366d2b1:      	movq	$0x1, 0x100(%rsp)
 366d2bd:      	movq	$0x1, 0x108(%rsp)
 366d2c9:      	callq	*0x1ad3ac1(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d2cf:      	movl	$0xe8, %edi
 366d2d4:      	movl	$0x8, %esi
 366d2d9:      	callq	*0x1ad3ab9(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d2df:      	testq	%rax, %rax
 366d2e2:      	je	0x366dbb6 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xa46>
 366d2e8:      	movq	%rax, %r15
 366d2eb:      	leaq	0x100(%rsp), %r13
 366d2f3:      	movl	$0xe8, %edx
 366d2f8:      	movq	%rax, %rdi
 366d2fb:      	movq	%r13, %rsi
 366d2fe:      	callq	*0x1ad3a34(%rip)        # 0x5140d38 <writev+0x5140d38>
 366d304:      	movq	%r15, 0x78(%rsp)
 366d309:      	lock
 366d30a:      	incq	(%r15)
 366d30d:      	jle	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
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
 366d38e:      	callq	0x3c141b0 <_RINvNtNtCs2AWtUsOyxgP_3std6thread9lifecycle15spawn_uncheckedNCNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full0uEB12_>
 366d393:      	movq	0x10(%rsp), %rcx
 366d398:      	movq	0x18(%rsp), %rax
 366d39d:      	testq	%rcx, %rcx
 366d3a0:      	je	0x366dcfb <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xb8b>
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
 366d3e9:      	jmp	0x366d3fa <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x28a>
 366d3eb:      	nopl	(%rax,%rax)
 366d3f0:      	xorl	%edi, %edi
 366d3f2:      	movl	$0xf4240, %esi          # imm = 0xF4240
 366d3f7:      	callq	*%r13
 366d3fa:      	decl	%r12d
 366d3fd:      	je	0x366d42f <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x2bf>
 366d3ff:      	movq	%r14, %rdi
 366d402:      	movq	%rbx, %rsi
 366d405:      	callq	0x3875ff0 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth4live>
 366d40a:      	movq	0x100(%rsp), %rsi
 366d412:      	cmpq	$-0x1, %rsi
 366d416:      	je	0x366d42f <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x2bf>
 366d418:      	testq	%rsi, %rsi
 366d41b:      	je	0x366d3f0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x280>
 366d41d:      	movq	0x108(%rsp), %rdi
 366d425:      	movl	$0x1, %edx
 366d42a:      	callq	*%r15
 366d42d:      	jmp	0x366d3f0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x280>
 366d42f:      	movq	$0x1, 0x3a8(%rsp)
 366d43b:      	movq	0x38(%rsp), %rax
 366d440:      	movq	%rax, 0x3b0(%rsp)
 366d448:      	callq	*0x1ad3942(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d44e:      	movl	$0x18, %edi
 366d453:      	movl	$0x8, %esi
 366d458:      	callq	*0x1ad393a(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d45e:      	testq	%rax, %rax
 366d461:      	je	0x366dbcb <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xa5b>
 366d467:      	movq	%rax, %rbx
 366d46a:      	callq	*0x1ad3920(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d470:      	movl	$0x1, %edi
 366d475:      	movl	$0x1, %esi
 366d47a:      	callq	*0x1ad3918(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d480:      	testq	%rax, %rax
 366d483:      	je	0x366dd27 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbb7>
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
 366d4f0:      	je	0x366dbe0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xa70>
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
 366d57f:      	je	0x366dbf5 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xa85>
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
 366d5cc:      	jne	0x366dcad <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xb3d>
 366d5d2:      	movq	%fs:(%r14), %rax
 366d5d6:      	movq	%fs:0x8(%r14), %rdx
 366d5db:      	leaq	0x1(%rax), %rcx
 366d5df:      	movq	%rcx, %fs:(%r14)
 366d5e3:      	movq	$0x1, 0x100(%rsp)
 366d5ef:      	movq	$0x1, 0x108(%rsp)
 366d5fb:      	movq	$-0x1, 0x110(%rsp)
 366d607:      	movups	0x1977e2a(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649>
 366d60e:      	movups	%xmm0, 0x188(%rsp)
 366d616:      	movups	0x1977e2b(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649+0x10>
 366d61d:      	movups	%xmm0, 0x198(%rsp)
 366d625:      	movq	%rax, 0x1a8(%rsp)
 366d62d:      	movq	%rdx, 0x1b0(%rsp)
 366d635:      	callq	*0x1ad3755(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d63b:      	movl	$0xb8, %edi
 366d640:      	movl	$0x8, %esi
 366d645:      	callq	*0x1ad374d(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d64b:      	testq	%rax, %rax
 366d64e:      	je	0x366dc12 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xaa2>
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
 366d6ba:      	je	0x366dc27 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xab7>
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
 366d737:      	callq	0x3a07a40 <_RNvXsX_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEENtNtCs4NRVxsYgnAr_4core7default7Default7defaultB1y_>
 366d73c:      	movq	%rax, 0x40(%rsp)
 366d741:      	callq	*0x1ad3649(%rip)        # 0x5140d90 <writev+0x5140d90>
 366d747:      	movl	$0x28, %edi
 366d74c:      	movl	$0x8, %esi
 366d751:      	callq	*0x1ad3641(%rip)        # 0x5140d98 <writev+0x5140d98>
 366d757:      	testq	%rax, %rax
 366d75a:      	movq	$-0x38, %r14
 366d761:      	je	0x366dc44 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xad4>
 366d767:      	movq	$0x1, (%rax)
 366d76e:      	movq	$0x1, 0x8(%rax)
 366d776:      	xorps	%xmm0, %xmm0
 366d779:      	movups	%xmm0, 0x10(%rax)
 366d77d:      	movq	$0x0, 0x20(%rax)
 366d785:      	movq	%rax, 0x48(%rsp)
 366d78a:      	cmpb	$0x1, %fs:0x10(%r14)
 366d790:      	jne	0x366dcc7 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xb57>
 366d796:      	movq	%fs:(%r14), %rax
 366d79a:      	movq	%fs:0x8(%r14), %rdx
 366d79f:      	leaq	0x1(%rax), %rcx
 366d7a3:      	movq	%rcx, %fs:(%r14)
 366d7a7:      	movups	0x1977c9a(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649+0x10>
 366d7ae:      	movups	%xmm0, 0x23(%rsp)
 366d7b3:      	movups	0x1977c7e(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649>
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
 366d838:      	je	0x366dc59 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xae9>
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
 366d896:      	callq	0x3eae160 <_RNvNtCs3pwlnhBXFtN_12memra_server9audio_api12shared_audio>
 366d89b:      	movq	%rax, 0x60(%rsp)
 366d8a0:      	callq	0x3c6ee10 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store12ttl_from_env>
 366d8a5:      	movq	%rax, %r13
 366d8a8:      	callq	0x3c6f0c0 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store18max_bytes_from_env>
 366d8ad:      	movq	%rax, %r14
 366d8b0:      	xorl	%edi, %edi
 366d8b2:      	callq	0x3c7c610 <_RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 366d8b7:      	testq	%rax, %rax
 366d8ba:      	je	0x366dd39 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbc9>
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
 366d976:      	je	0x366dc6e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xafe>
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
 366d9d7:      	leaq	0x19788f2(%rip), %rax   # 0x4fe62d0 <anon.60c0d423c0cc6e2470573bc6d977e900.615.llvm.10122458563676435649>
 366d9de:      	movq	%rax, 0xa8(%rsp)
 366d9e6:      	movq	$-0x38, %r14
 366d9ed:      	cmpb	$0x1, %fs:0x10(%r14)
 366d9f3:      	jne	0x366dce1 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xb71>
 366d9f9:      	movq	%fs:(%r14), %rax
 366d9fd:      	movq	%fs:0x8(%r14), %rdx
 366da02:      	leaq	0x1(%rax), %rcx
 366da06:      	movq	%rcx, %fs:(%r14)
 366da0a:      	movups	0x1977a37(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649+0x10>
 366da11:      	movups	%xmm0, 0x23(%rsp)
 366da16:      	movups	0x1977a1b(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649>
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
 366da9b:      	je	0x366dc83 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xb13>
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
 366dbc6:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dbcb:      	movl	$0x8, %edi
 366dbd0:      	movl	$0x18, %esi
 366dbd5:      	callq	*0x1ad3385(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dbdb:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dbe0:      	movl	$0x8, %edi
 366dbe5:      	movl	$0x28, %esi
 366dbea:      	callq	*0x1ad3370(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dbf0:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dbf5:      	leaq	0x110(%rsp), %rbx
 366dbfd:      	movl	$0x8, %edi
 366dc02:      	movl	$0x40, %esi
 366dc07:      	callq	*0x1ad3353(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc0d:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc12:      	movl	$0x8, %edi
 366dc17:      	movl	$0xb8, %esi
 366dc1c:      	callq	*0x1ad333e(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc22:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc27:      	leaq	0x120(%rsp), %rbx
 366dc2f:      	movl	$0x8, %edi
 366dc34:      	movl	$0x28, %esi
 366dc39:      	callq	*0x1ad3321(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc3f:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc44:      	movl	$0x8, %edi
 366dc49:      	movl	$0x28, %esi
 366dc4e:      	callq	*0x1ad330c(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc54:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc59:      	movl	$0x8, %edi
 366dc5e:      	movl	$0x48, %esi
 366dc63:      	callq	*0x1ad32f7(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc69:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc6e:      	movl	$0x8, %edi
 366dc73:      	movl	$0x68, %esi
 366dc78:      	callq	*0x1ad32e2(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc7e:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc83:      	movl	$0x8, %edi
 366dc88:      	movl	$0x48, %esi
 366dc8d:      	callq	*0x1ad32cd(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dc93:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dc98:      	movl	$0x80, %edi
 366dc9d:      	movl	$0x200, %esi            # imm = 0x200
 366dca2:      	callq	*0x1ad32b8(%rip)        # 0x5140f60 <writev+0x5140f60>
 366dca8:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dcad:      	callq	*0x1ad34e5(%rip)        # 0x5141198 <writev+0x5141198>
 366dcb3:      	movq	%rax, %fs:(%r14)
 366dcb7:      	movq	%rdx, %fs:0x8(%r14)
 366dcbc:      	movb	$0x1, %fs:0x10(%r14)
 366dcc2:      	jmp	0x366d5db <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x46b>
 366dcc7:      	callq	*0x1ad34cb(%rip)        # 0x5141198 <writev+0x5141198>
 366dccd:      	movq	%rax, %fs:(%r14)
 366dcd1:      	movq	%rdx, %fs:0x8(%r14)
 366dcd6:      	movb	$0x1, %fs:0x10(%r14)
 366dcdc:      	jmp	0x366d79f <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x62f>
 366dce1:      	callq	*0x1ad34b1(%rip)        # 0x5141198 <writev+0x5141198>
 366dce7:      	movq	%rax, %fs:(%r14)
 366dceb:      	movq	%rdx, %fs:0x8(%r14)
 366dcf0:      	movb	$0x1, %fs:0x10(%r14)
 366dcf6:      	jmp	0x366da02 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0x892>
 366dcfb:      	movq	%rax, 0x10(%rsp)
 366dd00:      	leaq	-0x3266de2(%rip), %rdi  # 0x406f25 <anon.60c0d423c0cc6e2470573bc6d977e900.3554.llvm.10122458563676435649+0x242>
 366dd07:      	leaq	0x198183a(%rip), %rcx   # 0x4fef548 <anon.60c0d423c0cc6e2470573bc6d977e900.3560.llvm.10122458563676435649>
 366dd0e:      	leaq	0x1981f83(%rip), %r8    # 0x4fefc98 <anon.60c0d423c0cc6e2470573bc6d977e900.3581.llvm.10122458563676435649+0x4b0>
 366dd15:      	leaq	0x10(%rsp), %rdx
 366dd1a:      	movl	$0x16, %esi
 366dd1f:      	callq	*0x1ad3173(%rip)        # 0x5140e98 <writev+0x5140e98>
 366dd25:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dd27:      	movl	$0x1, %edi
 366dd2c:      	movl	$0x1, %esi
 366dd31:      	callq	*0x1ad3051(%rip)        # 0x5140d88 <writev+0x5140d88>
 366dd37:      	jmp	0x366dd46 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xbd6>
 366dd39:      	leaq	0x19a9d90(%rip), %rdi   # 0x5017ad0 <anon.090efebfaa215158d04f118ce3d95888.2.llvm.9611048873305048036>
 366dd40:      	callq	*0x1ad442a(%rip)        # 0x5142170 <writev+0x5142170>
 366dd46:      	ud2
 366dd48:      	movq	%rax, %r15
 366dd4b:      	jmp	0x366ddf1 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xc81>
 366dd50:      	movq	%rax, %r15
 366dd53:      	jmp	0x366de9e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd2e>
 366dd58:      	movq	%rax, %r15
 366dd5b:      	jmp	0x366dfa5 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe35>
 366dd60:      	movq	%rax, %r15
 366dd63:      	jmp	0x366de39 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xcc9>
 366dd68:      	movq	%rax, %r15
 366dd6b:      	movb	$0x1, %r14b
 366dd6e:      	jmp	0x366ded9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd69>
 366dd73:      	movq	%rax, %r15
 366dd76:      	movl	$0x18, %esi
 366dd7b:      	movl	$0x8, %edx
 366dd80:      	movq	%rbx, %rdi
 366dd83:      	callq	*0x1ad2fb7(%rip)        # 0x5140d40 <writev+0x5140d40>
 366dd89:      	jmp	0x366e013 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xea3>
 366dd8e:      	movq	%rax, %r15
 366dd91:      	leaq	0x10(%rsp), %rdi
 366dd96:      	callq	0x3550de0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtNtCs2AWtUsOyxgP_3std2io5error5ErrorECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 366dd9b:      	jmp	0x366ddb5 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xc45>
 366dd9d:      	callq	*0x1ad3015(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dda3:      	movq	%rax, %r15
 366dda6:      	jmp	0x366e057 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xee7>
 366ddab:      	jmp	0x366ddb2 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xc42>
 366ddad:      	movq	%rax, %r15
 366ddb0:      	jmp	0x366de24 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xcb4>
 366ddb2:      	movq	%rax, %r15
 366ddb5:      	movb	$0x1, %r14b
 366ddb8:      	movb	$0x1, %bl
 366ddba:      	movq	0x8(%rsp), %rax
 366ddbf:      	jmp	0x366e032 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xec2>
 366ddc4:      	movq	%rax, %r15
 366ddc7:      	leaq	0x100(%rsp), %rdi
 366ddcf:      	callq	0x3c04360 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync4mpmc7counter7CounterINtNtBG_4list7ChannelNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdEEEB1R_.llvm.12140772379724168186>
 366ddd4:      	movb	$0x1, %bl
 366ddd6:      	jmp	0x366e07b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xf0b>
 366dddb:      	callq	*0x1ad2fd7(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dde1:      	movq	%rax, %r15
 366dde4:      	leaq	0x118(%rsp), %rdi
 366ddec:      	callq	0x392d540 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server15background_jobs7ControlEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366ddf1:      	movq	0xa0(%rsp), %rax
 366ddf9:      	lock
 366ddfa:      	decq	(%rax)
 366ddfd:      	jne	0x366de24 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xcb4>
 366ddff:      	leaq	0xa0(%rsp), %rdi
 366de07:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 366de0c:      	jmp	0x366de24 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xcb4>
 366de0e:      	callq	*0x1ad2fa4(%rip)        # 0x5140db8 <writev+0x5140db8>
 366de14:      	movq	%rax, %r15
 366de17:      	leaq	0x118(%rsp), %rdi
 366de1f:      	callq	0x392dff0 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366de24:      	movq	0x60(%rsp), %rax
 366de29:      	lock
 366de2a:      	decq	(%rax)
 366de2d:      	jne	0x366de39 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xcc9>
 366de2f:      	leaq	0x60(%rsp), %rdi
 366de34:      	callq	0x39bbe90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCsdMwdNqnNPrU_11memra_lanes12audio_stream14AudioSchedulerEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366de39:      	movq	0xb0(%rsp), %rax
 366de41:      	testq	%rax, %rax
 366de44:      	je	0x366de59 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xce9>
 366de46:      	lock
 366de47:      	decq	(%rax)
 366de4a:      	jne	0x366de59 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xce9>
 366de4c:      	leaq	0xb0(%rsp), %rdi
 366de54:      	callq	0x39bd090 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server8darklane10BgJobStateE9drop_slowBJ_>
 366de59:      	movq	0x58(%rsp), %rax
 366de5e:      	lock
 366de5f:      	decq	(%rax)
 366de62:      	jne	0x366de6e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xcfe>
 366de64:      	leaq	0x58(%rsp), %rdi
 366de69:      	callq	0x39bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 366de6e:      	movq	0x50(%rsp), %rax
 366de73:      	lock
 366de74:      	decq	(%rax)
 366de77:      	jne	0x366de83 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd13>
 366de79:      	leaq	0x50(%rsp), %rdi
 366de7e:      	callq	0x39bb500 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexINtNtNtNtBO_11collections4hash3map7HashMapNtNtB7_6string6StringjEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366de83:      	xorl	%r14d, %r14d
 366de86:      	jmp	0x366dea1 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd31>
 366de88:      	callq	*0x1ad2f2a(%rip)        # 0x5140db8 <writev+0x5140db8>
 366de8e:      	movq	%rax, %r15
 366de91:      	leaq	0x118(%rsp), %rdi
 366de99:      	callq	0x392e960 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringyEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropCs3pwlnhBXFtN_12memra_server>
 366de9e:      	movb	$0x1, %r14b
 366dea1:      	movq	0x48(%rsp), %rax
 366dea6:      	lock
 366dea7:      	decq	(%rax)
 366deaa:      	jne	0x366dec4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd54>
 366deac:      	leaq	0x48(%rsp), %rdi
 366deb1:      	callq	0x39b9f90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6worker15EventQueueStateE9drop_slowBJ_>
 366deb6:      	jmp	0x366dec4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd54>
 366deb8:      	callq	*0x1ad2efa(%rip)        # 0x5140db8 <writev+0x5140db8>
 366debe:      	movq	%rax, %r15
 366dec1:      	movb	$0x1, %r14b
 366dec4:      	movq	0x40(%rsp), %rax
 366dec9:      	lock
 366deca:      	decq	(%rax)
 366decd:      	jne	0x366ded9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd69>
 366decf:      	leaq	0x40(%rsp), %rdi
 366ded4:      	callq	0x39bb8d0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex5MutexNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEE9drop_slowB1y_>
 366ded9:      	movq	0xe8(%rsp), %rax
 366dee1:      	testq	%rax, %rax
 366dee4:      	je	0x366defa <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd8a>
 366dee6:      	lock
 366dee7:      	decq	(%rax)
 366deea:      	jne	0x366defa <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xd8a>
 366deec:      	leaq	0xe8(%rsp), %rdi
 366def4:      	callq	*0x1ad40ae(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 366defa:      	movq	0xd0(%rsp), %rax
 366df02:      	testq	%rax, %rax
 366df05:      	je	0x366df1b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xdab>
 366df07:      	lock
 366df08:      	decq	(%rax)
 366df0b:      	jne	0x366df1b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xdab>
 366df0d:      	leaq	0xd0(%rsp), %rdi
 366df15:      	callq	*0x1ad408d(%rip)        # 0x5141fa8 <writev+0x5141fa8>
 366df1b:      	movq	0x98(%rsp), %rax
 366df23:      	testq	%rax, %rax
 366df26:      	je	0x366df3b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xdcb>
 366df28:      	lock
 366df29:      	decq	(%rax)
 366df2c:      	jne	0x366df3b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xdcb>
 366df2e:      	leaq	0x98(%rsp), %rdi
 366df36:      	callq	0x39bb080 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringIBw_NtCscIVK9LJ3de3_15memra_tokenizer9TokenizerEEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366df3b:      	movq	0x3b8(%rsp), %rax
 366df43:      	testq	%rax, %rax
 366df46:      	je	0x366df5b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xdeb>
 366df48:      	lock
 366df49:      	decq	(%rax)
 366df4c:      	jne	0x366df5b <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xdeb>
 366df4e:      	leaq	0x3b8(%rsp), %rdi
 366df56:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 366df5b:      	movq	0x90(%rsp), %rax
 366df63:      	lock
 366df64:      	decq	(%rax)
 366df67:      	jne	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe38>
 366df69:      	leaq	0x90(%rsp), %rdi
 366df71:      	callq	0x39bc0a0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockIBw_NtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEE9drop_slowB1C_>
 366df76:      	jmp	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe38>
 366df78:      	movq	%rax, %r15
 366df7b:      	lock
 366df7c:      	decq	(%r13)
 366df80:      	movb	$0x1, %r14b
 366df83:      	jne	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe38>
 366df85:      	movq	%rbx, %rdi
 366df88:      	callq	0x39bc740 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetE9drop_slowBH_>
 366df8d:      	jmp	0x366dfa8 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe38>
 366df8f:      	callq	*0x1ad2e23(%rip)        # 0x5140db8 <writev+0x5140db8>
 366df95:      	movq	%rax, %r15
 366df98:      	leaq	0x100(%rsp), %rdi
 366dfa0:      	callq	0x352f670 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCscdodAO9FK5_5alloc4sync8ArcInnerNtCs3pwlnhBXFtN_12memra_server16ModelMetadataSetEEB1f_>
 366dfa5:      	movb	$0x1, %r14b
 366dfa8:      	movq	0x88(%rsp), %rax
 366dfb0:      	lock
 366dfb1:      	decq	(%rax)
 366dfb4:      	jne	0x366dfd9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe69>
 366dfb6:      	leaq	0x88(%rsp), %rdi
 366dfbe:      	callq	0x39bb0f0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtNtNtCs2AWtUsOyxgP_3std11collections4hash3map7HashMapNtNtB7_6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEE9drop_slowB1Z_>
 366dfc3:      	jmp	0x366dfd9 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe69>
 366dfc5:      	callq	*0x1ad2ded(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dfcb:      	movq	%rax, %r15
 366dfce:      	movq	%rbx, %rdi
 366dfd1:      	callq	0x392de80 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
 366dfd6:      	movb	$0x1, %r14b
 366dfd9:      	movq	0x80(%rsp), %rax
 366dfe1:      	lock
 366dfe2:      	decq	(%rax)
 366dfe5:      	jne	0x366dff4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xe84>
 366dfe7:      	leaq	0x80(%rsp), %rdi
 366dfef:      	callq	0x39ba050 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcINtNtB7_3vec3VecNtNtB7_6string6StringEE9drop_slowCs3pwlnhBXFtN_12memra_server>
 366dff4:      	xorl	%ebx, %ebx
 366dff6:      	jmp	0x366e018 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xea8>
 366dff8:      	callq	*0x1ad2dba(%rip)        # 0x5140db8 <writev+0x5140db8>
 366dffe:      	movq	%rax, %r15
 366e001:      	leaq	0x100(%rsp), %rdi
 366e009:      	callq	0x352f3d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCscdodAO9FK5_5alloc4sync8ArcInnerINtNtBG_3vec3VecNtNtBG_6string6StringEEECs3pwlnhBXFtN_12memra_server>
 366e00e:      	jmp	0x366e013 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xea3>
 366e010:      	movq	%rax, %r15
 366e013:      	movb	$0x1, %bl
 366e015:      	movb	$0x1, %r14b
 366e018:      	leaq	0x3a8(%rsp), %rdi
 366e020:      	callq	0x3c723b0 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 366e025:      	testb	%r14b, %r14b
 366e028:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xf20>
 366e02a:      	movq	0x78(%rsp), %rax
 366e02f:      	xorl	%r14d, %r14d
 366e032:      	lock
 366e033:      	decq	(%rax)
 366e036:      	jne	0x366e042 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xed2>
 366e038:      	leaq	0x78(%rsp), %rdi
 366e03d:      	callq	0x39bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 366e042:      	testb	%r14b, %r14b
 366e045:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xf20>
 366e047:      	xorl	%r14d, %r14d
 366e04a:      	jmp	0x366e069 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xef9>
 366e04c:      	movq	%rax, %r15
 366e04f:      	movq	%r13, %rdi
 366e052:      	callq	0x384d3d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthEBF_.llvm.1341170882515377341>
 366e057:      	movb	$0x1, %bl
 366e059:      	leaq	0x398(%rsp), %rdi
 366e061:      	callq	0x3c79320 <_RNvXsi_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_8ReceiverNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBU_>
 366e066:      	movb	$0x1, %r14b
 366e069:      	leaq	0x388(%rsp), %rdi
 366e071:      	callq	0x3c723b0 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server6worker3CmdENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 366e076:      	testb	%r14b, %r14b
 366e079:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xf20>
 366e07b:      	cmpq	$-0x1, 0x70(%rsp)
 366e081:      	je	0x366e090 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xf20>
 366e083:      	leaq	0x378(%rsp), %rdi
 366e08b:      	callq	0x3c71d30 <_RNvXs4_NtNtCs2AWtUsOyxgP_3std4sync4mpmcINtB5_6SenderNtNtCs3pwlnhBXFtN_12memra_server5tests9WorkerSawENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropBS_>
 366e090:      	testb	%bl, %bl
 366e092:      	je	0x366e09e <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649+0xf2e>
 366e094:      	movq	0x68(%rsp), %rdi
 366e099:      	callq	0x392de80 <_RNvXsg_NtCsgQfI1edjipl_9hashbrown3rawINtB5_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server6worker9ModelCapsEENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1u_>
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
 366f76f:      	movl	0x1ade9bb(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891+0x8>
 366f775:      	testl	%eax, %eax
 366f777:      	jne	0x366f950 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x250>
 366f77d:      	movq	0x1ade9a4(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
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
 366f7b3:      	callq	0x3ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
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
 366f87d:      	movl	0x1ade8ad(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891+0x8>
 366f883:      	testl	%eax, %eax
 366f885:      	jne	0x366f8d4 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0x1d4>
 366f887:      	movq	0x1ade89a(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 366f88e:      	movq	%rbx, %rdi
 366f891:      	movq	%rbp, %rsi
 366f894:      	xorl	%edx, %edx
 366f896:      	movq	%r12, %rcx
 366f899:      	leaq	0x20(%rsp), %r8
 366f89e:      	leaq	0x1ae3e13(%rip), %rax   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 366f8a5:      	pushq	%rax
 366f8a6:      	leaq	0x1ae3df3(%rip), %rax   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 366f8ad:      	pushq	%rax
 366f8ae:      	callq	0x3ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
 366f8b3:      	addq	$0x10, %rsp
 366f8b7:      	cmpl	$-0x1, 0x38(%rsp)
 366f8bc:      	je	0x366f7d0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0xd0>
 366f8c2:      	leaq	0x38(%rsp), %r14
 366f8c7:      	movq	%r14, %rdi
 366f8ca:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 366f8cf:      	jmp	0x366f7df <_RNvNtCs3pwlnhBXFtN_12memra_server5tests38reserve_interactive_through_contention+0xdf>
 366f8d4:      	leaq	0x1ade84d(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 366f8db:      	callq	0x3b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
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
 366f923:      	leaq	0x396336(%rip), %rax    # 0x3a05c60 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtRReNtB6_7Display3fmtCs3pwlnhBXFtN_12memra_server>
 366f92a:      	movq	%rax, 0xd0(%rsp)
 366f932:      	leaq	-0x32dd295(%rip), %rdi  # 0x3926a4 <anon.c030ccd08983dd522ef199ffd40d4bd6.208.llvm.13931728001802268293+0x24>
 366f939:      	leaq	0x1980698(%rip), %rdx   # 0x4feffd8 <anon.60c0d423c0cc6e2470573bc6d977e900.3581.llvm.10122458563676435649+0x7f0>
 366f940:      	leaq	0xc8(%rsp), %rsi
 366f948:      	callq	*0x1ad1532(%rip)        # 0x5140e80 <writev+0x5140e80>
 366f94e:      	ud2
 366f950:      	leaq	0x1ade7d1(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 366f957:      	callq	0x3b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
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
 367e07a:      	movups	0x19673b7(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649>
 367e081:      	movaps	%xmm0, 0x40(%rsp)
 367e086:      	movups	0x19673bb(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649+0x10>
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
 367e0bd:      	callq	0x366d170 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649>
 367e0c2:      	addq	$0x10, %rsp
 367e0c6:      	movl	0x1ad00bc(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x18>
 367e0cc:      	testl	%eax, %eax
 367e0ce:      	jne	0x367e427 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x477>
 367e0d4:      	movq	0x1ad00a5(%rip), %rbx   # 0x514e180 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x10>
 367e0db:      	movl	0x1acffff(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x10>
 367e0e1:      	testl	%eax, %eax
 367e0e3:      	jne	0x367e438 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x488>
 367e0e9:      	movq	0x1acffe0(%rip), %rcx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 367e0f0:      	movq	0x1acffe1(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x8>
 367e0f7:      	movq	%rbx, %rdx
 367e0fa:      	shrq	$0x3e, %rdx
 367e0fe:      	jne	0x367e449 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x499>
 367e104:      	shlq	$0x2, %rbx
 367e108:      	testq	%rcx, %rcx
 367e10b:      	cmovneq	%rax, %rbx
 367e10f:      	movq	%rbx, 0x1ad559a(%rip)   # 0x51536b0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS+0x10>
 367e116:      	movq	$0x0, 0x1ad557f(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 367e121:      	movl	0x1ad0061(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x18>
 367e127:      	testl	%eax, %eax
 367e129:      	jne	0x367e455 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4a5>
 367e12f:      	movq	0x1ad003a(%rip), %rax   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
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
 367e17a:      	movl	0x1acffb0(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891+0x8>
 367e180:      	testl	%eax, %eax
 367e182:      	jne	0x367e466 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x4b6>
 367e188:      	movq	0x1acff99(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 367e18f:      	leaq	0x1ad5522(%rip), %r15   # 0x51536b8 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker14PENDING_ADMITS>
 367e196:      	leaq	0x1ad5503(%rip), %r14   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 367e19d:      	leaq	0x108(%rsp), %rdi
 367e1a5:      	leaq	0x198(%rsp), %rsi
 367e1ad:      	movq	%rsp, %rcx
 367e1b0:      	leaq	0x40(%rsp), %r8
 367e1b5:      	xorl	%edx, %edx
 367e1b7:      	pushq	%r15
 367e1b9:      	pushq	%r14
 367e1bb:      	callq	0x3ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
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
 367e262:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 367e267:      	jmp	0x367e273 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x2c3>
 367e269:      	leaq	0x48(%rsp), %rdi
 367e26e:      	callq	0x354c0c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>
 367e273:      	movl	0x1acff0f(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x18>
 367e279:      	testl	%eax, %eax
 367e27b:      	jne	0x367e4d2 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x522>
 367e281:      	movq	0x1acfef8(%rip), %rax   # 0x514e180 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x10>
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
 367e2e7:      	movl	0x1acfe43(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891+0x8>
 367e2ed:      	testl	%eax, %eax
 367e2ef:      	jne	0x367e4e3 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x533>
 367e2f5:      	movq	0x1acfe2c(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 367e2fc:      	leaq	0x40(%rsp), %rdi
 367e301:      	leaq	0x198(%rsp), %rsi
 367e309:      	leaq	0xe8(%rsp), %rcx
 367e311:      	leaq	0xd0(%rsp), %r8
 367e319:      	movl	$0x2, %edx
 367e31e:      	pushq	%r15
 367e320:      	pushq	%r14
 367e322:      	callq	0x3ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
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
 367e372:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 367e377:      	movq	0x18(%rsp), %rax
 367e37c:      	testq	%rax, %rax
 367e37f:      	je	0x367e38f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x3df>
 367e381:      	lock
 367e382:      	decq	(%rax)
 367e385:      	jne	0x367e38f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x3df>
 367e387:      	movq	%rbx, %rdi
 367e38a:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 367e38f:      	movq	$0x0, 0x1ad5316(%rip)   # 0x51536b0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS+0x10>
 367e39a:      	leaq	0x198(%rsp), %rdi
 367e3a2:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649>
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
 367e427:      	leaq	0x1acfd42(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 367e42e:      	callq	0x3b2b258 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 367e433:      	jmp	0x367e0d4 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x124>
 367e438:      	leaq	0x1acfc91(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 367e43f:      	callq	0x3b2b3b1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 367e444:      	jmp	0x367e0e9 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x139>
 367e449:      	movq	$-0x1, %rbx
 367e450:      	jmp	0x367e108 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x158>
 367e455:      	leaq	0x1acfd14(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 367e45c:      	callq	0x3b2b258 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 367e461:      	jmp	0x367e12f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x17f>
 367e466:      	leaq	0x1acfcbb(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 367e46d:      	callq	0x3b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 367e472:      	jmp	0x367e188 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x1d8>
 367e477:      	leaq	0x40(%rsp), %r14
 367e47c:      	leaq	0x108(%rsp), %rsi
 367e484:      	movl	$0x90, %edx
 367e489:      	movq	%r14, %rdi
 367e48c:      	callq	*0x1ac28a6(%rip)        # 0x5140d38 <writev+0x5140d38>
 367e492:      	leaq	-0x3277029(%rip), %rdi  # 0x407470 <anon.60c0d423c0cc6e2470573bc6d977e900.3771.llvm.10122458563676435649+0x254>
 367e499:      	leaq	0x1971288(%rip), %rcx   # 0x4fef728 <anon.60c0d423c0cc6e2470573bc6d977e900.3562.llvm.10122458563676435649+0x1a0>
 367e4a0:      	leaq	0x1972109(%rip), %r8    # 0x4ff05b0 <anon.60c0d423c0cc6e2470573bc6d977e900.3878.llvm.10122458563676435649+0x18>
 367e4a7:      	movl	$0x3a, %esi
 367e4ac:      	movq	%r14, %rdx
 367e4af:      	callq	*0x1ac29e3(%rip)        # 0x5140e98 <writev+0x5140e98>
 367e4b5:      	jmp	0x367e517 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x567>
 367e4b7:      	leaq	-0x3276f7d(%rip), %rdi  # 0x407541 <anon.60c0d423c0cc6e2470573bc6d977e900.3771.llvm.10122458563676435649+0x325>
 367e4be:      	leaq	0x197211b(%rip), %rdx   # 0x4ff05e0 <anon.60c0d423c0cc6e2470573bc6d977e900.3878.llvm.10122458563676435649+0x48>
 367e4c5:      	movl	$0x139, %esi            # imm = 0x139
 367e4ca:      	callq	*0x1ac29b0(%rip)        # 0x5140e80 <writev+0x5140e80>
 367e4d0:      	jmp	0x367e517 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x567>
 367e4d2:      	leaq	0x1acfc97(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 367e4d9:      	callq	0x3b2b258 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 367e4de:      	jmp	0x367e281 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x2d1>
 367e4e3:      	leaq	0x1acfc3e(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 367e4ea:      	callq	0x3b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 367e4ef:      	jmp	0x367e2f5 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x345>
 367e4f4:      	leaq	0x40(%rsp), %rdi
 367e4f9:      	callq	0x352d6c0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardTINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEReEEEBZ_>
 367e4fe:      	leaq	-0x327705b(%rip), %rdi  # 0x4074aa <anon.60c0d423c0cc6e2470573bc6d977e900.3771.llvm.10122458563676435649+0x28e>
 367e505:      	leaq	0x19720bc(%rip), %rdx   # 0x4ff05c8 <anon.60c0d423c0cc6e2470573bc6d977e900.3878.llvm.10122458563676435649+0x30>
 367e50c:      	movl	$0x97, %esi
 367e511:      	callq	*0x1ac29f1(%rip)        # 0x5140f08 <writev+0x5140f08>
 367e517:      	ud2
 367e519:      	movq	%rax, %r14
 367e51c:      	movzbl	%bl, %esi
 367e51f:      	leaq	0x1ad519a(%rip), %rdi   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 367e526:      	callq	0x35312d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
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
 367e560:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 367e565:      	movq	0x18(%rsp), %rax
 367e56a:      	testq	%rax, %rax
 367e56d:      	je	0x367e57d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5cd>
 367e56f:      	lock
 367e570:      	decq	(%rax)
 367e573:      	jne	0x367e57d <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_38admission_reservations_are_lane_scoped+0x5cd>
 367e575:      	movq	%rbx, %rdi
 367e578:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 367e57d:      	movq	$0x0, 0x1ad5128(%rip)   # 0x51536b0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS+0x10>
 367e588:      	leaq	0x198(%rsp), %rdi
 367e590:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649>
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
 3710500:      	movups	0x18d4f31(%rip), %xmm0  # 0x4fe5438 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649>
 3710507:      	movaps	%xmm0, 0x50(%rsp)
 371050c:      	movups	0x18d4f35(%rip), %xmm0  # 0x4fe5448 <anon.60c0d423c0cc6e2470573bc6d977e900.12.llvm.10122458563676435649+0x10>
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
 3710543:      	callq	0x366d170 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests22fake_worker_state_full.llvm.10122458563676435649>
 3710548:      	addq	$0x10, %rsp
 371054c:      	movl	0x1a3dc36(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x18>
 3710552:      	testl	%eax, %eax
 3710554:      	jne	0x3710899 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x469>
 371055a:      	movq	0x1a3dc0f(%rip), %rbx   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 3710561:      	movl	0x1a3db79(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x10>
 3710567:      	testl	%eax, %eax
 3710569:      	jne	0x37108aa <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x47a>
 371056f:      	movq	0x1a3db5a(%rip), %rcx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 3710576:      	movq	0x1a3db5b(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x8>
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
 3710610:      	movl	0x1a3db1a(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891+0x8>
 3710616:      	testl	%eax, %eax
 3710618:      	jne	0x37108e5 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x4b5>
 371061e:      	movq	0x1a3db03(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
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
 3710659:      	callq	0x3ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
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
 3710754:      	movl	0x1a3d9d6(%rip), %eax   # 0x514e130 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891+0x8>
 371075a:      	testl	%eax, %eax
 371075c:      	jne	0x3710991 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x561>
 3710762:      	movq	0x1a3d9bf(%rip), %r9    # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 3710769:      	leaq	0x50(%rsp), %rdi
 371076e:      	leaq	0x1b0(%rsp), %rsi
 3710776:      	leaq	0x30(%rsp), %rcx
 371077b:      	leaq	0x120(%rsp), %r8
 3710783:      	xorl	%edx, %edx
 3710785:      	pushq	%r13
 3710787:      	pushq	%r12
 3710789:      	callq	0x3ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>
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
 37107d2:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 37107d7:      	movq	$0x0, 0x1a42ebe(%rip)   # 0x51536a0 <_RNvNtCs3pwlnhBXFtN_12memra_server6worker22ADMISSION_RESERVATIONS>
 37107e2:      	leaq	0x1b0(%rsp), %rdi
 37107ea:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649>
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
 3710811:      	leaq	-0x31b4c95(%rip), %rdi  # 0x55bb83 <anon.60c0d423c0cc6e2470573bc6d977e900.6219.llvm.10122458563676435649+0x2625>
 3710818:      	leaq	0x18e7331(%rip), %rdx   # 0x4ff7b50 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x1b0>
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
 3710899:      	leaq	0x1a3d8d0(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 37108a0:      	callq	0x3b2b258 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 37108a5:      	jmp	0x371055a <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x12a>
 37108aa:      	leaq	0x1a3d81f(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 37108b1:      	callq	0x3b2b3b1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 37108b6:      	jmp	0x371056f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x13f>
 37108bb:      	movq	$-0x1, %r15
 37108c2:      	jmp	0x3710592 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x162>
 37108c7:      	leaq	-0x31b4da3(%rip), %rdi  # 0x55bb2b <anon.60c0d423c0cc6e2470573bc6d977e900.6219.llvm.10122458563676435649+0x25cd>
 37108ce:      	leaq	0x18e721b(%rip), %rdx   # 0x4ff7af0 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x150>
 37108d5:      	movl	$0x5f, %esi
 37108da:      	callq	*0x1a305a0(%rip)        # 0x5140e80 <writev+0x5140e80>
 37108e0:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 37108e5:      	leaq	0x1a3d83c(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 37108ec:      	callq	0x3b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 37108f1:      	jmp	0x371061e <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x1ee>
 37108f6:      	leaq	0x50(%rsp), %rbx
 37108fb:      	leaq	0x120(%rsp), %rsi
 3710903:      	movl	$0x90, %edx
 3710908:      	movq	%rbx, %rdi
 371090b:      	callq	*0x1a30427(%rip)        # 0x5140d38 <writev+0x5140d38>
 3710911:      	leaq	-0x31b4dbe(%rip), %rdi  # 0x55bb5a <anon.60c0d423c0cc6e2470573bc6d977e900.6219.llvm.10122458563676435649+0x25fc>
 3710918:      	leaq	0x18dee09(%rip), %rcx   # 0x4fef728 <anon.60c0d423c0cc6e2470573bc6d977e900.3562.llvm.10122458563676435649+0x1a0>
 371091f:      	leaq	0x18e71e2(%rip), %r8    # 0x4ff7b08 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x168>
 3710926:      	movl	$0x29, %esi
 371092b:      	movq	%rbx, %rdx
 371092e:      	callq	*0x1a30564(%rip)        # 0x5140e98 <writev+0x5140e98>
 3710934:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 3710936:      	leaq	0x18e7243(%rip), %r9    # 0x4ff7b80 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x1e0>
 371093d:      	leaq	-0x33866e4(%rip), %rdx  # 0x38a260 <anon.ea9ff54ec891c523b22d7beb433c6217.3025.llvm.10671892735646583555>
 3710944:      	jmp	0x3710952 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x522>
 3710946:      	leaq	0x18e71d3(%rip), %r9    # 0x4ff7b20 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x180>
 371094d:      	leaq	0x8(%rsp), %rdx
 3710952:      	xorl	%edi, %edi
 3710954:      	movq	%r14, %rsi
 3710957:      	xorl	%ecx, %ecx
 3710959:      	callq	*0x1a30671(%rip)        # 0x5140fd0 <writev+0x5140fd0>
 371095f:      	jmp	0x371098f <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x55f>
 3710961:      	leaq	0x18e7200(%rip), %r9    # 0x4ff7b68 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x1c8>
 3710968:      	leaq	-0x33864d7(%rip), %rdx  # 0x38a498 <anon.c030ccd08983dd522ef199ffd40d4bd6.616.llvm.13931728001802268293>
 371096f:      	leaq	0x50(%rsp), %rbx
 3710974:      	jmp	0x3710982 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x552>
 3710976:      	leaq	0x18e71bb(%rip), %r9    # 0x4ff7b38 <anon.60c0d423c0cc6e2470573bc6d977e900.6475.llvm.10122458563676435649+0x198>
 371097d:      	leaq	0x50(%rsp), %rdx
 3710982:      	xorl	%edi, %edi
 3710984:      	movq	%rbx, %rsi
 3710987:      	xorl	%ecx, %ecx
 3710989:      	callq	*0x1a30641(%rip)        # 0x5140fd0 <writev+0x5140fd0>
 371098f:      	ud2
 3710991:      	leaq	0x1a3d790(%rip), %rdi   # 0x514e128 <_RNvNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s1S.llvm.4933824046006920891>
 3710998:      	callq	0x3b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>
 371099d:      	jmp	0x3710762 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x332>
 37109a2:      	movq	%rax, %rbx
 37109a5:      	movzbl	%bpl, %esi
 37109a9:      	leaq	0x1a42d10(%rip), %rdi   # 0x51536c0 <_RNvNtCs3pwlnhBXFtN_12memra_server5tests10DRAIN_LOCK>
 37109b0:      	callq	0x35312d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
 37109b5:      	jmp	0x3710a24 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5f4>
 37109b7:      	callq	*0x1a303fb(%rip)        # 0x5140db8 <writev+0x5140db8>
 37109bd:      	movq	%rax, %rbx
 37109c0:      	jmp	0x3710a1a <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5ea>
 37109c2:      	jmp	0x37109f0 <_RNvNtCs3pwlnhBXFtN_12memra_server5testss_62pending_admission_reservation_is_atomic_and_rolls_back_on_drop+0x5c0>
 37109c4:      	movq	%rax, %rbx
 37109c7:      	leaq	0x50(%rsp), %rdi
 37109cc:      	callq	0x352d8b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCs1S6izGDSMYD_4http8response8ResponseNtNtCsaSG9NyffgI5_9axum_core4body4BodyEECs3pwlnhBXFtN_12memra_server.llvm.10122458563676435649>
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
 3710a15:      	callq	0x354c4e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server8AppStateEBD_.llvm.10122458563676435649>
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

000000000384d590 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_>:
 384d590:      	pushq	%rbx
 384d591:      	movq	%rdi, %rbx
 384d594:      	movq	0x10(%rdi), %rsi
 384d598:      	testq	%rsi, %rsi
 384d59b:      	je	0x384d5ac <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_+0x1c>
 384d59d:      	movq	0x18(%rbx), %rdi
 384d5a1:      	movl	$0x1, %edx
 384d5a6:      	callq	*0x18f3794(%rip)        # 0x5140d40 <writev+0x5140d40>
 384d5ac:      	movq	0x28(%rbx), %rsi
 384d5b0:      	cmpq	$-0x1, %rsi
 384d5b4:      	je	0x384d5cb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_+0x3b>
 384d5b6:      	testq	%rsi, %rsi
 384d5b9:      	je	0x384d5cb <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_+0x3b>
 384d5bb:      	movq	0x30(%rbx), %rdi
 384d5bf:      	movl	$0x1, %edx
 384d5c4:      	popq	%rbx
 384d5c5:      	jmpq	*0x18f3775(%rip)        # 0x5140d40 <writev+0x5140d40>
 384d5cb:      	popq	%rbx
 384d5cc:      	retq
 384d5cd:      	int3
 384d5ce:      	int3
 384d5cf:      	int3

0000000003875410 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms>:
 3875410:      	pushq	%r14
 3875412:      	pushq	%rbx
 3875413:      	subq	$0x228, %rsp            # imm = 0x228
 387541a:      	movq	%rdi, %rbx
 387541d:      	leaq	0x68(%rsp), %rdi
 3875422:      	callq	0x388f890 <_RNvXs1_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealthNtNtCs4NRVxsYgnAr_4core7default7Default7default>
 3875427:      	movl	0x138(%rsp), %eax
 387542e:      	movups	0x68(%rsp), %xmm0
 3875433:      	movups	0x78(%rsp), %xmm1
 3875438:      	movups	0x88(%rsp), %xmm2
 3875440:      	movups	0x98(%rsp), %xmm3
 3875448:      	movaps	%xmm0, (%rsp)
 387544c:      	movaps	%xmm1, 0x10(%rsp)
 3875451:      	movaps	%xmm2, 0x20(%rsp)
 3875456:      	movaps	%xmm3, 0x30(%rsp)
 387545b:      	movups	0xa8(%rsp), %xmm0
 3875463:      	movaps	%xmm0, 0x40(%rsp)
 3875468:      	movups	0xb8(%rsp), %xmm0
 3875470:      	movaps	%xmm0, 0x50(%rsp)
 3875475:      	movq	0x130(%rsp), %rcx
 387547d:      	movq	%rcx, 0x218(%rsp)
 3875485:      	movups	0x120(%rsp), %xmm0
 387548d:      	movups	%xmm0, 0x208(%rsp)
 3875495:      	movups	0x110(%rsp), %xmm0
 387549d:      	movups	%xmm0, 0x1f8(%rsp)
 38754a5:      	movq	$0x1, 0x140(%rsp)
 38754b1:      	movq	$0x1, 0x148(%rsp)
 38754bd:      	movaps	(%rsp), %xmm0
 38754c1:      	movaps	0x10(%rsp), %xmm1
 38754c6:      	movaps	0x20(%rsp), %xmm2
 38754cb:      	movaps	0x30(%rsp), %xmm3
 38754d0:      	movups	%xmm0, 0x150(%rsp)
 38754d8:      	movups	%xmm1, 0x160(%rsp)
 38754e0:      	movups	%xmm2, 0x170(%rsp)
 38754e8:      	movups	%xmm3, 0x180(%rsp)
 38754f0:      	movaps	0x40(%rsp), %xmm0
 38754f5:      	movups	%xmm0, 0x190(%rsp)
 38754fd:      	movaps	0x50(%rsp), %xmm0
 3875502:      	movups	%xmm0, 0x1a0(%rsp)
 387550a:      	movq	$0x0, 0x1b0(%rsp)
 3875516:      	movups	0xd8(%rsp), %xmm0
 387551e:      	movups	%xmm0, 0x1c0(%rsp)
 3875526:      	movups	0xe8(%rsp), %xmm0
 387552e:      	movups	%xmm0, 0x1d0(%rsp)
 3875536:      	movups	0xf8(%rsp), %xmm0
 387553e:      	movups	%xmm0, 0x1e0(%rsp)
 3875546:      	movq	%rbx, 0x1f0(%rsp)
 387554e:      	movl	%eax, 0x220(%rsp)
 3875555:      	movl	0x13c(%rsp), %eax
 387555c:      	movl	%eax, 0x224(%rsp)
 3875563:      	callq	*0x18cb827(%rip)        # 0x5140d90 <writev+0x5140d90>
 3875569:      	movl	$0xe8, %edi
 387556e:      	movl	$0x8, %esi
 3875573:      	callq	*0x18cb81f(%rip)        # 0x5140d98 <writev+0x5140d98>
 3875579:      	testq	%rax, %rax
 387557c:      	je	0x38755c5 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1b5>
 387557e:      	movq	%rax, %rbx
 3875581:      	leaq	0x140(%rsp), %rsi
 3875589:      	movl	$0xe8, %edx
 387558e:      	movq	%rax, %rdi
 3875591:      	callq	*0x18cb7a1(%rip)        # 0x5140d38 <writev+0x5140d38>
 3875597:      	movq	0xc8(%rsp), %rax
 387559f:      	testq	%rax, %rax
 38755a2:      	je	0x38755b7 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1a7>
 38755a4:      	lock
 38755a5:      	decq	(%rax)
 38755a8:      	jne	0x38755b7 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1a7>
 38755aa:      	leaq	0xc8(%rsp), %rdi
 38755b2:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 38755b7:      	movq	%rbx, %rax
 38755ba:      	addq	$0x228, %rsp            # imm = 0x228
 38755c1:      	popq	%rbx
 38755c2:      	popq	%r14
 38755c4:      	retq
 38755c5:      	leaq	0x150(%rsp), %r14
 38755cd:      	movl	$0x8, %edi
 38755d2:      	movl	$0xe8, %esi
 38755d7:      	callq	*0x18cb983(%rip)        # 0x5140f60 <writev+0x5140f60>
 38755dd:      	ud2
 38755df:      	movq	%rax, %rbx
 38755e2:      	movq	%r14, %rdi
 38755e5:      	callq	0x384d3d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthEBF_.llvm.1341170882515377341>
 38755ea:      	movq	0xc8(%rsp), %rax
 38755f2:      	testq	%rax, %rax
 38755f5:      	je	0x387560a <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1fa>
 38755f7:      	lock
 38755f8:      	decq	(%rax)
 38755fb:      	jne	0x387560a <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms+0x1fa>
 38755fd:      	leaq	0xc8(%rsp), %rdi
 3875605:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 387560a:      	movq	%rbx, %rdi
 387560d:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3875612:      	callq	*0x18cb7a0(%rip)        # 0x5140db8 <writev+0x5140db8>
 3875618:      	callq	*0x18cb79a(%rip)        # 0x5140db8 <writev+0x5140db8>
 387561e:      	int3
 387561f:      	int3

0000000003875620 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route>:
 3875620:      	pushq	%rbp
 3875621:      	pushq	%r15
 3875623:      	pushq	%r14
 3875625:      	pushq	%r13
 3875627:      	pushq	%r12
 3875629:      	pushq	%rbx
 387562a:      	subq	$0xc8, %rsp
 3875631:      	movq	%rdi, %rbx
 3875634:      	movq	%rsi, 0x18(%rsp)
 3875639:      	movq	%rdx, 0x20(%rsp)
 387563e:      	leaq	0x38(%rsp), %rdi
 3875643:      	callq	0x3878ad0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth3new>
 3875648:      	movq	$0x1, 0x28(%rsp)
 3875651:      	movq	$0x1, 0x30(%rsp)
 387565a:      	callq	*0x18cb730(%rip)        # 0x5140d90 <writev+0x5140d90>
 3875660:      	movl	$0xa0, %edi
 3875665:      	movl	$0x8, %esi
 387566a:      	callq	*0x18cb728(%rip)        # 0x5140d98 <writev+0x5140d98>
 3875670:      	testq	%rax, %rax
 3875673:      	je	0x3875784 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x164>
 3875679:      	movq	%rax, %r14
 387567c:      	leaq	0x28(%rsp), %rsi
 3875681:      	movl	$0xa0, %edx
 3875686:      	movq	%rax, %rdi
 3875689:      	callq	*0x18cb6a9(%rip)        # 0x5140d38 <writev+0x5140d38>
 387568f:      	movq	%r14, 0x8(%rsp)
 3875694:      	leaq	0xa8(%rbx), %r14
 387569b:      	movl	$0x3fffffff, %ecx       # imm = 0x3FFFFFFF
 38756a0:      	xorl	%eax, %eax
 38756a2:      	lock
 38756a3:      	cmpxchgl	%ecx, 0xa8(%rbx)
 38756aa:      	jne	0x3875796 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x176>
 38756b0:      	movq	0x18cb9b1(%rip), %rax   # 0x5141068 <writev+0x5141068>
 38756b7:      	movq	(%rax), %rax
 38756ba:      	shlq	%rax
 38756bd:      	testq	%rax, %rax
 38756c0:      	jne	0x38757a4 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x184>
 38756c6:      	xorl	%ebp, %ebp
 38756c8:      	movzbl	0xb0(%rbx), %eax
 38756cf:      	movq	%r14, 0x28(%rsp)
 38756d4:      	movb	%bpl, 0x30(%rsp)
 38756d9:      	leaq	0xb8(%rbx), %r12
 38756e0:      	leaq	0x18(%rsp), %rsi
 38756e5:      	movq	%r12, %rdi
 38756e8:      	callq	0x3adc8b0 <_RINvMs_NtCscdodAO9FK5_5alloc3vecINtB5_3VecINtNtB7_4sync3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthEE6retainNCNvMs2_BX_NtBX_12WorkerHealth14register_routes_0EBZ_>
 38756ed:      	movq	0x8(%rsp), %r15
 38756f2:      	lock
 38756f3:      	incq	(%r15)
 38756f6:      	jle	0x3875794 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x174>
 38756fc:      	movq	%r15, 0x10(%rsp)
 3875701:      	movq	0xc8(%rbx), %r13
 3875708:      	cmpq	0xb8(%rbx), %r13
 387570f:      	jne	0x3875719 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0xf9>
 3875711:      	movq	%r12, %rdi
 3875714:      	callq	0x39b3570 <_RNvMs3_NtCscdodAO9FK5_5alloc7raw_vecINtB5_6RawVecQNtCskf97wSN6mZ4_8memra_kv5CacheE8grow_oneCs3pwlnhBXFtN_12memra_server>
 3875719:      	movq	0xc0(%rbx), %rax
 3875720:      	movq	%r15, (%rax,%r13,8)
 3875724:      	incq	%r13
 3875727:      	movq	%r13, 0xc8(%rbx)
 387572e:      	testb	%bpl, %bpl
 3875731:      	jne	0x387574c <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x12c>
 3875733:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 387573d:      	movq	0x18cb924(%rip), %rcx   # 0x5141068 <writev+0x5141068>
 3875744:      	movq	(%rcx), %rcx
 3875747:      	testq	%rax, %rcx
 387574a:      	jne	0x38757b5 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x195>
 387574c:      	movl	$0xc0000001, %esi       # imm = 0xC0000001
 3875751:      	lock
 3875752:      	xaddl	%esi, (%r14)
 3875756:      	addl	$0xc0000001, %esi       # imm = 0xC0000001
 387575c:      	cmpl	$0x40000000, %esi       # imm = 0x40000000
 3875762:      	jae	0x3875779 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x159>
 3875764:      	movq	%r15, %rax
 3875767:      	addq	$0xc8, %rsp
 387576e:      	popq	%rbx
 387576f:      	popq	%r12
 3875771:      	popq	%r13
 3875773:      	popq	%r14
 3875775:      	popq	%r15
 3875777:      	popq	%rbp
 3875778:      	retq
 3875779:      	movq	%r14, %rdi
 387577c:      	callq	*0x18cc7c6(%rip)        # 0x5141f48 <writev+0x5141f48>
 3875782:      	jmp	0x3875764 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x144>
 3875784:      	movl	$0x8, %edi
 3875789:      	movl	$0xa0, %esi
 387578e:      	callq	*0x18cb7cc(%rip)        # 0x5140f60 <writev+0x5140f60>
 3875794:      	ud2
 3875796:      	movq	%r14, %rdi
 3875799:      	callq	*0x18cca79(%rip)        # 0x5142218 <writev+0x5142218>
 387579f:      	jmp	0x38756b0 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x90>
 38757a4:      	callq	*0x18cb8ce(%rip)        # 0x5141078 <writev+0x5141078>
 38757aa:      	movl	%eax, %ebp
 38757ac:      	xorb	$0x1, %bpl
 38757b0:      	jmp	0x38756c8 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0xa8>
 38757b5:      	callq	*0x18cb8bd(%rip)        # 0x5141078 <writev+0x5141078>
 38757bb:      	testb	%al, %al
 38757bd:      	jne	0x387574c <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x12c>
 38757bf:      	movb	$0x1, 0xb0(%rbx)
 38757c6:      	jmp	0x387574c <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x12c>
 38757c8:      	movq	%rax, %rbx
 38757cb:      	jmp	0x38757f5 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1d5>
 38757cd:      	movq	%rax, %rbx
 38757d0:      	lock
 38757d1:      	decq	(%r15)
 38757d4:      	jne	0x38757eb <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1cb>
 38757d6:      	leaq	0x10(%rsp), %rdi
 38757db:      	callq	0x39bcf00 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 38757e0:      	jmp	0x38757eb <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1cb>
 38757e2:      	callq	*0x18cb5d0(%rip)        # 0x5140db8 <writev+0x5140db8>
 38757e8:      	movq	%rax, %rbx
 38757eb:      	leaq	0x28(%rsp), %rdi
 38757f0:      	callq	0x3ba89a0 <_RNvXsi_NtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlockINtB5_16RwLockWriteGuardNtNtCs3pwlnhBXFtN_12memra_server4auth5StateENtNtNtCs4NRVxsYgnAr_4core3ops4drop4Drop4dropB1e_>
 38757f5:      	movq	0x8(%rsp), %rax
 38757fa:      	lock
 38757fb:      	decq	(%rax)
 38757fe:      	jne	0x387581f <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1ff>
 3875800:      	leaq	0x8(%rsp), %rdi
 3875805:      	callq	0x39bcf00 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 387580a:      	jmp	0x387581f <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route+0x1ff>
 387580c:      	callq	*0x18cb5a6(%rip)        # 0x5140db8 <writev+0x5140db8>
 3875812:      	movq	%rax, %rbx
 3875815:      	leaq	0x28(%rsp), %rdi
 387581a:      	callq	0x384c940 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtCscdodAO9FK5_5alloc4sync8ArcInnerNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthEEB1h_>
 387581f:      	movq	%rbx, %rdi
 3875822:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3875827:      	callq	*0x18cb58b(%rip)        # 0x5140db8 <writev+0x5140db8>
 387582d:      	int3
 387582e:      	int3
 387582f:      	int3

00000000038786a0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>:
 38786a0:      	pushq	%rbp
 38786a1:      	pushq	%r15
 38786a3:      	pushq	%r14
 38786a5:      	pushq	%r12
 38786a7:      	pushq	%rbx
 38786a8:      	subq	$0x10, %rsp
 38786ac:      	movq	%rdi, %r14
 38786af:      	leaq	0x60(%rdi), %rbx
 38786b3:      	movl	$0x1, %ecx
 38786b8:      	xorl	%eax, %eax
 38786ba:      	lock
 38786bb:      	cmpxchgl	%ecx, 0x60(%rdi)
 38786bf:      	jne	0x387879b <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xfb>
 38786c5:      	movq	0x18c899c(%rip), %r15   # 0x5141068 <writev+0x5141068>
 38786cc:      	movq	(%r15), %rax
 38786cf:      	shlq	%rax
 38786d2:      	testq	%rax, %rax
 38786d5:      	jne	0x38787a9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x109>
 38786db:      	xorl	%ebp, %ebp
 38786dd:      	movabsq	$0x7fffffffffffffff, %r12 # imm = 0x7FFFFFFFFFFFFFFF
 38786e7:      	movzbl	0x64(%r14), %eax
 38786ec:      	movq	0x58(%r14), %rax
 38786f0:      	testq	%rax, %rax
 38786f3:      	je	0x3878707 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x67>
 38786f5:      	leaq	-0x1(%rax), %rcx
 38786f9:      	lock
 38786fa:      	cmpxchgq	%rcx, 0x58(%r14)
 38786ff:      	jne	0x38786f0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x50>
 3878701:      	cmpq	$0x1, %rax
 3878705:      	ja	0x3878778 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xd8>
 3878707:      	movzbl	0x88(%r14), %eax
 387870f:      	cmpb	$0x3, %al
 3878711:      	je	0x3878778 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xd8>
 3878713:      	cmpb	$0x0, %fs:-0x1c90
 387871c:      	je	0x3878729 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x89>
 387871e:      	movq	%fs:-0x1c88, %rax
 3878727:      	jmp	0x387876c <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xcc>
 3878729:      	movl	0x18d57c1(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x10>
 387872f:      	testl	%eax, %eax
 3878731:      	jne	0x38787d6 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x136>
 3878737:      	movq	0x18d57a2(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 387873e:      	movl	0x18d57a4(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x8>
 3878744:      	movq	%rax, (%rsp)
 3878748:      	movl	%ecx, 0x8(%rsp)
 387874c:      	movq	%rsp, %rdi
 387874f:      	callq	*0x18c8763(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 3878755:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 387875c:      	movl	%edx, %eax
 387875e:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 3878765:      	shrq	$0x32, %rax
 3878769:      	addq	%rcx, %rax
 387876c:      	movq	%rax, 0x28(%r14)
 3878770:      	movb	$0x1, 0x88(%r14)
 3878778:      	testb	%bpl, %bpl
 387877b:      	jne	0x3878785 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xe5>
 387877d:      	movq	(%r15), %rax
 3878780:      	testq	%r12, %rax
 3878783:      	jne	0x38787c5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x125>
 3878785:      	xorl	%eax, %eax
 3878787:      	xchgl	%eax, (%rbx)
 3878789:      	cmpl	$0x2, %eax
 387878c:      	je	0x38787ba <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x11a>
 387878e:      	addq	$0x10, %rsp
 3878792:      	popq	%rbx
 3878793:      	popq	%r12
 3878795:      	popq	%r14
 3878797:      	popq	%r15
 3878799:      	popq	%rbp
 387879a:      	retq
 387879b:      	movq	%rbx, %rdi
 387879e:      	callq	*0x18c8c64(%rip)        # 0x5141408 <writev+0x5141408>
 38787a4:      	jmp	0x38786c5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x25>
 38787a9:      	callq	*0x18c88c9(%rip)        # 0x5141078 <writev+0x5141078>
 38787af:      	movl	%eax, %ebp
 38787b1:      	xorb	$0x1, %bpl
 38787b5:      	jmp	0x38786dd <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x3d>
 38787ba:      	movq	%rbx, %rdi
 38787bd:      	callq	*0x18c88ad(%rip)        # 0x5141070 <writev+0x5141070>
 38787c3:      	jmp	0x387878e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xee>
 38787c5:      	callq	*0x18c88ad(%rip)        # 0x5141078 <writev+0x5141078>
 38787cb:      	testb	%al, %al
 38787cd:      	jne	0x3878785 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xe5>
 38787cf:      	movb	$0x1, 0x64(%r14)
 38787d4:      	jmp	0x3878785 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0xe5>
 38787d6:      	leaq	0x18d5703(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 38787dd:      	callq	0x3b2b88b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 38787e2:      	jmp	0x3878737 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request+0x97>
 38787e7:      	movq	%rax, %r14
 38787ea:      	movzbl	%bpl, %esi
 38787ee:      	movq	%rbx, %rdi
 38787f1:      	callq	0x384cf70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.1341170882515377341>
 38787f6:      	movq	%r14, %rdi
 38787f9:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 38787fe:      	callq	*0x18c85b4(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878804:      	int3
 3878805:      	int3
 3878806:      	int3
 3878807:      	int3
 3878808:      	int3
 3878809:      	int3
 387880a:      	int3
 387880b:      	int3
 387880c:      	int3
 387880d:      	int3
 387880e:      	int3
 387880f:      	int3

0000000003878810 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>:
 3878810:      	pushq	%rbp
 3878811:      	pushq	%r15
 3878813:      	pushq	%r14
 3878815:      	pushq	%rbx
 3878816:      	subq	$0x18, %rsp
 387881a:      	movq	%rdi, %r14
 387881d:      	lock
 387881e:      	incq	0x48(%rdi)
 3878822:      	leaq	0x60(%rdi), %rbx
 3878826:      	movl	$0x1, %ecx
 387882b:      	xorl	%eax, %eax
 387882d:      	lock
 387882e:      	cmpxchgl	%ecx, 0x60(%rdi)
 3878832:      	jne	0x38788f9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xe9>
 3878838:      	movq	0x18c8829(%rip), %r15   # 0x5141068 <writev+0x5141068>
 387883f:      	movq	(%r15), %rax
 3878842:      	shlq	%rax
 3878845:      	testq	%rax, %rax
 3878848:      	jne	0x3878907 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xf7>
 387884e:      	xorl	%ebp, %ebp
 3878850:      	movzbl	0x64(%r14), %eax
 3878855:      	lock
 3878856:      	incq	0x58(%r14)
 387885a:      	movzbl	0x88(%r14), %eax
 3878862:      	cmpb	$0x3, %al
 3878864:      	je	0x38788ce <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xbe>
 3878866:      	cmpb	$0x0, %fs:-0x1c90
 387886f:      	je	0x387887c <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x6c>
 3878871:      	movq	%fs:-0x1c88, %rax
 387887a:      	jmp	0x38788c2 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xb2>
 387887c:      	movl	0x18d566e(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x10>
 3878882:      	testl	%eax, %eax
 3878884:      	jne	0x3878934 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x124>
 387888a:      	movq	0x18d564f(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 3878891:      	movl	0x18d5651(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x8>
 3878897:      	movq	%rax, 0x8(%rsp)
 387889c:      	movl	%ecx, 0x10(%rsp)
 38788a0:      	leaq	0x8(%rsp), %rdi
 38788a5:      	callq	*0x18c860d(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 38788ab:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 38788b2:      	movl	%edx, %eax
 38788b4:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 38788bb:      	shrq	$0x32, %rax
 38788bf:      	addq	%rcx, %rax
 38788c2:      	movq	%rax, 0x28(%r14)
 38788c6:      	movb	$0x2, 0x88(%r14)
 38788ce:      	testb	%bpl, %bpl
 38788d1:      	jne	0x38788e5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xd5>
 38788d3:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 38788dd:      	movq	(%r15), %rcx
 38788e0:      	testq	%rax, %rcx
 38788e3:      	jne	0x3878923 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x113>
 38788e5:      	xorl	%eax, %eax
 38788e7:      	xchgl	%eax, (%rbx)
 38788e9:      	cmpl	$0x2, %eax
 38788ec:      	je	0x3878918 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x108>
 38788ee:      	addq	$0x18, %rsp
 38788f2:      	popq	%rbx
 38788f3:      	popq	%r14
 38788f5:      	popq	%r15
 38788f7:      	popq	%rbp
 38788f8:      	retq
 38788f9:      	movq	%rbx, %rdi
 38788fc:      	callq	*0x18c8b06(%rip)        # 0x5141408 <writev+0x5141408>
 3878902:      	jmp	0x3878838 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x28>
 3878907:      	callq	*0x18c876b(%rip)        # 0x5141078 <writev+0x5141078>
 387890d:      	movl	%eax, %ebp
 387890f:      	xorb	$0x1, %bpl
 3878913:      	jmp	0x3878850 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x40>
 3878918:      	movq	%rbx, %rdi
 387891b:      	callq	*0x18c874f(%rip)        # 0x5141070 <writev+0x5141070>
 3878921:      	jmp	0x38788ee <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xde>
 3878923:      	callq	*0x18c874f(%rip)        # 0x5141078 <writev+0x5141078>
 3878929:      	testb	%al, %al
 387892b:      	jne	0x38788e5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xd5>
 387892d:      	movb	$0x1, 0x64(%r14)
 3878932:      	jmp	0x38788e5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0xd5>
 3878934:      	leaq	0x18d55a5(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 387893b:      	callq	0x3b2b88b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 3878940:      	jmp	0x387888a <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request+0x7a>
 3878945:      	movq	%rax, %r14
 3878948:      	movzbl	%bpl, %esi
 387894c:      	movq	%rbx, %rdi
 387894f:      	callq	0x384cf70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.1341170882515377341>
 3878954:      	movq	%r14, %rdi
 3878957:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 387895c:      	callq	*0x18c8456(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878962:      	int3
 3878963:      	int3
 3878964:      	int3
 3878965:      	int3
 3878966:      	int3
 3878967:      	int3
 3878968:      	int3
 3878969:      	int3
 387896a:      	int3
 387896b:      	int3
 387896c:      	int3
 387896d:      	int3
 387896e:      	int3
 387896f:      	int3

0000000003878970 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free>:
 3878970:      	pushq	%rbp
 3878971:      	pushq	%r15
 3878973:      	pushq	%r14
 3878975:      	pushq	%rbx
 3878976:      	subq	$0x18, %rsp
 387897a:      	movq	%rdi, %r14
 387897d:      	leaq	0x60(%rdi), %rbx
 3878981:      	movl	$0x1, %ecx
 3878986:      	xorl	%eax, %eax
 3878988:      	lock
 3878989:      	cmpxchgl	%ecx, 0x60(%rdi)
 387898d:      	jne	0x3878a58 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xe8>
 3878993:      	movq	0x18c86ce(%rip), %r15   # 0x5141068 <writev+0x5141068>
 387899a:      	movq	(%r15), %rax
 387899d:      	shlq	%rax
 38789a0:      	testq	%rax, %rax
 38789a3:      	jne	0x3878a66 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xf6>
 38789a9:      	xorl	%ebp, %ebp
 38789ab:      	movzbl	0x64(%r14), %eax
 38789b0:      	movq	0x58(%r14), %rax
 38789b4:      	testq	%rax, %rax
 38789b7:      	jne	0x3878a2d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xbd>
 38789b9:      	movzbl	0x88(%r14), %eax
 38789c1:      	cmpb	$0x3, %al
 38789c3:      	je	0x3878a2d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xbd>
 38789c5:      	cmpb	$0x0, %fs:-0x1c90
 38789ce:      	je	0x38789db <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x6b>
 38789d0:      	movq	%fs:-0x1c88, %rax
 38789d9:      	jmp	0x3878a21 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xb1>
 38789db:      	movl	0x18d550f(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x10>
 38789e1:      	testl	%eax, %eax
 38789e3:      	jne	0x3878aa1 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x131>
 38789e9:      	movq	0x18d54f0(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 38789f0:      	movl	0x18d54f2(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x8>
 38789f6:      	movq	%rax, 0x8(%rsp)
 38789fb:      	movl	%ecx, 0x10(%rsp)
 38789ff:      	leaq	0x8(%rsp), %rdi
 3878a04:      	callq	*0x18c84ae(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 3878a0a:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 3878a11:      	movl	%edx, %eax
 3878a13:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 3878a1a:      	shrq	$0x32, %rax
 3878a1e:      	addq	%rcx, %rax
 3878a21:      	movq	%rax, 0x28(%r14)
 3878a25:      	movb	$0x1, 0x88(%r14)
 3878a2d:      	testb	%bpl, %bpl
 3878a30:      	jne	0x3878a44 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xd4>
 3878a32:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3878a3c:      	movq	(%r15), %rcx
 3878a3f:      	testq	%rax, %rcx
 3878a42:      	jne	0x3878a90 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x120>
 3878a44:      	xorl	%eax, %eax
 3878a46:      	xchgl	%eax, (%rbx)
 3878a48:      	cmpl	$0x2, %eax
 3878a4b:      	je	0x3878a85 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x115>
 3878a4d:      	addq	$0x18, %rsp
 3878a51:      	popq	%rbx
 3878a52:      	popq	%r14
 3878a54:      	popq	%r15
 3878a56:      	popq	%rbp
 3878a57:      	retq
 3878a58:      	movq	%rbx, %rdi
 3878a5b:      	callq	*0x18c89a7(%rip)        # 0x5141408 <writev+0x5141408>
 3878a61:      	jmp	0x3878993 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x23>
 3878a66:      	callq	*0x18c860c(%rip)        # 0x5141078 <writev+0x5141078>
 3878a6c:      	movl	%eax, %ebp
 3878a6e:      	xorb	$0x1, %bpl
 3878a72:      	movzbl	0x64(%r14), %eax
 3878a77:      	movq	0x58(%r14), %rax
 3878a7b:      	testq	%rax, %rax
 3878a7e:      	jne	0x3878a2d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xbd>
 3878a80:      	jmp	0x38789b9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x49>
 3878a85:      	movq	%rbx, %rdi
 3878a88:      	callq	*0x18c85e2(%rip)        # 0x5141070 <writev+0x5141070>
 3878a8e:      	jmp	0x3878a4d <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xdd>
 3878a90:      	callq	*0x18c85e2(%rip)        # 0x5141078 <writev+0x5141078>
 3878a96:      	testb	%al, %al
 3878a98:      	jne	0x3878a44 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xd4>
 3878a9a:      	movb	$0x1, 0x64(%r14)
 3878a9f:      	jmp	0x3878a44 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0xd4>
 3878aa1:      	leaq	0x18d5438(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 3878aa8:      	callq	0x3b2b88b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 3878aad:      	jmp	0x38789e9 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free+0x79>
 3878ab2:      	movq	%rax, %r14
 3878ab5:      	movzbl	%bpl, %esi
 3878ab9:      	movq	%rbx, %rdi
 3878abc:      	callq	0x384cf70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuarduEECs3pwlnhBXFtN_12memra_server.llvm.1341170882515377341>
 3878ac1:      	movq	%r14, %rdi
 3878ac4:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3878ac9:      	callq	*0x18c82e9(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878acf:      	int3

0000000003878c40 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341>:
 3878c40:      	pushq	%rbp
 3878c41:      	pushq	%r15
 3878c43:      	pushq	%r14
 3878c45:      	pushq	%r13
 3878c47:      	pushq	%r12
 3878c49:      	pushq	%rbx
 3878c4a:      	subq	$0xb8, %rsp
 3878c51:      	movq	%rsi, %r14
 3878c54:      	movq	%rdi, %rbx
 3878c57:      	movzbl	0x88(%rsi), %eax
 3878c5e:      	movb	%al, 0x7(%rsp)
 3878c62:      	cmpb	$0x0, %fs:-0x1c90
 3878c6b:      	je	0x3878c78 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x38>
 3878c6d:      	movq	%fs:-0x1c88, %r13
 3878c76:      	jmp	0x3878cbe <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x7e>
 3878c78:      	movl	0x18d5272(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x10>
 3878c7e:      	testl	%eax, %eax
 3878c80:      	jne	0x3878f28 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x2e8>
 3878c86:      	movq	0x18d5253(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 3878c8d:      	movl	0x18d5255(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x8>
 3878c93:      	movq	%rax, 0x70(%rsp)
 3878c98:      	movl	%ecx, 0x78(%rsp)
 3878c9c:      	leaq	0x70(%rsp), %rdi
 3878ca1:      	callq	*0x18c8211(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 3878ca7:      	imulq	$0x3e8, %rax, %rax      # imm = 0x3E8
 3878cae:      	movl	%edx, %ecx
 3878cb0:      	imulq	$0x431bde83, %rcx, %r13 # imm = 0x431BDE83
 3878cb7:      	shrq	$0x32, %r13
 3878cbb:      	addq	%rax, %r13
 3878cbe:      	leaq	0x88(%rsp), %rdi
 3878cc6:      	movq	%r14, %rsi
 3878cc9:      	callq	*0x18c82c9(%rip)        # 0x5140f98 <writev+0x5140f98>
 3878ccf:      	movq	0x20(%r14), %rax
 3878cd3:      	movq	%rax, 0x68(%rsp)
 3878cd8:      	movq	0x28(%r14), %rax
 3878cdc:      	movq	%rax, 0x60(%rsp)
 3878ce1:      	movq	0x30(%r14), %rax
 3878ce5:      	testq	%rax, %rax
 3878ce8:      	je	0x3878d0a <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0xca>
 3878cea:      	decq	%rax
 3878ced:      	xorl	%edx, %edx
 3878cef:      	movq	%r13, %rcx
 3878cf2:      	subq	%rax, %rcx
 3878cf5:      	cmovaeq	%rcx, %rdx
 3878cf9:      	movq	%rdx, 0x20(%rsp)
 3878cfe:      	movl	$0x1, %eax
 3878d03:      	movq	%rax, 0x18(%rsp)
 3878d08:      	jmp	0x3878d13 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0xd3>
 3878d0a:      	movq	$0x0, 0x18(%rsp)
 3878d13:      	movq	0x28(%r14), %rax
 3878d17:      	movq	%rax, 0x38(%rsp)
 3878d1c:      	movq	0x30(%r14), %rbp
 3878d20:      	testq	%rbp, %rbp
 3878d23:      	je	0x3878d38 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0xf8>
 3878d25:      	leaq	-0x1(%rbp), %rax
 3878d29:      	xorl	%r15d, %r15d
 3878d2c:      	movq	%r13, %rcx
 3878d2f:      	subq	%rax, %rcx
 3878d32:      	cmovaeq	%rcx, %r15
 3878d36:      	jmp	0x3878d38 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0xf8>
 3878d38:      	movq	0x38(%r14), %rax
 3878d3c:      	movq	%rax, 0x58(%rsp)
 3878d41:      	movq	0x40(%r14), %rax
 3878d45:      	movq	%rax, 0x50(%rsp)
 3878d4a:      	movq	0x48(%r14), %rax
 3878d4e:      	movq	%rax, 0x48(%rsp)
 3878d53:      	movq	0x50(%r14), %rax
 3878d57:      	movq	%rax, 0x40(%rsp)
 3878d5c:      	movq	0x18(%r14), %rax
 3878d60:      	movq	0x30(%rax), %rcx
 3878d64:      	movq	%rcx, 0x30(%rsp)
 3878d69:      	movq	0x38(%rax), %r12
 3878d6d:      	movq	0x40(%rax), %rax
 3878d71:      	movq	%rax, 0x28(%rsp)
 3878d76:      	movq	$-0x1, 0x8(%rsp)
 3878d7f:      	cmpb	$0x3, 0x7(%rsp)
 3878d84:      	jne	0x3878e5e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x21e>
 3878d8a:      	movl	$0x1, %ecx
 3878d8f:      	xorl	%eax, %eax
 3878d91:      	lock
 3878d92:      	cmpxchgl	%ecx, 0x68(%r14)
 3878d97:      	jne	0x3878e5e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x21e>
 3878d9d:      	movq	0x18c82c4(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3878da4:      	movq	(%rax), %rax
 3878da7:      	shlq	%rax
 3878daa:      	testq	%rax, %rax
 3878dad:      	jne	0x3878f39 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x2f9>
 3878db3:      	xorl	%ecx, %ecx
 3878db5:      	leaq	0x68(%r14), %rax
 3878db9:      	movq	%rax, 0x10(%rsp)
 3878dbe:      	movabsq	$0x7fffffffffffffff, %rdx # imm = 0x7FFFFFFFFFFFFFFF
 3878dc8:      	movzbl	0x6c(%r14), %eax
 3878dcd:      	testb	%al, %al
 3878dcf:      	je	0x3878dfe <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1be>
 3878dd1:      	testb	%cl, %cl
 3878dd3:      	jne	0x3878de8 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1a8>
 3878dd5:      	movq	0x18c828c(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3878ddc:      	movq	(%rax), %rax
 3878ddf:      	testq	%rdx, %rax
 3878de2:      	jne	0x3878f54 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x314>
 3878de8:      	xorl	%eax, %eax
 3878dea:      	movq	0x10(%rsp), %rdi
 3878def:      	xchgl	%eax, (%rdi)
 3878df1:      	cmpl	$0x2, %eax
 3878df4:      	jne	0x3878e5e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x21e>
 3878df6:      	callq	*0x18c8274(%rip)        # 0x5141070 <writev+0x5141070>
 3878dfc:      	jmp	0x3878e5e <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x21e>
 3878dfe:      	movl	%ecx, 0x8(%rsp)
 3878e02:      	leaq	0x70(%r14), %rsi
 3878e06:      	leaq	0x70(%rsp), %rdi
 3878e0b:      	callq	*0x18c8187(%rip)        # 0x5140f98 <writev+0x5140f98>
 3878e11:      	cmpb	$0x0, 0x8(%rsp)
 3878e16:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3878e20:      	jne	0x3878e35 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1f5>
 3878e22:      	movq	0x18c823f(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3878e29:      	movq	(%rax), %rax
 3878e2c:      	testq	%rcx, %rax
 3878e2f:      	jne	0x3878f6c <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x32c>
 3878e35:      	xorl	%eax, %eax
 3878e37:      	movq	0x10(%rsp), %rdi
 3878e3c:      	xchgl	%eax, (%rdi)
 3878e3e:      	cmpl	$0x2, %eax
 3878e41:      	je	0x3878f49 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x309>
 3878e47:      	movq	0x70(%rsp), %rax
 3878e4c:      	movq	%rax, 0x8(%rsp)
 3878e51:      	movups	0x78(%rsp), %xmm0
 3878e56:      	movaps	%xmm0, 0xa0(%rsp)
 3878e5e:      	addq	0x30(%rsp), %r12
 3878e63:      	addq	0x28(%rsp), %r12
 3878e68:      	xorl	%eax, %eax
 3878e6a:      	movq	%r13, %rcx
 3878e6d:      	subq	0x38(%rsp), %rcx
 3878e72:      	cmovbq	%rax, %rcx
 3878e76:      	cmpq	%rcx, %r15
 3878e79:      	cmovaeq	%rcx, %r15
 3878e7d:      	testq	%rbp, %rbp
 3878e80:      	cmoveq	%rcx, %r15
 3878e84:      	movq	%r13, %rcx
 3878e87:      	subq	0x60(%rsp), %rcx
 3878e8c:      	cmovbq	%rax, %rcx
 3878e90:      	subq	0x68(%rsp), %r13
 3878e95:      	cmovbq	%rax, %r13
 3878e99:      	movq	0x98(%rsp), %rax
 3878ea1:      	movq	%rax, 0x20(%rbx)
 3878ea5:      	movups	0x88(%rsp), %xmm0
 3878ead:      	movups	%xmm0, 0x10(%rbx)
 3878eb1:      	movzbl	0x7(%rsp), %eax
 3878eb6:      	movb	%al, 0x80(%rbx)
 3878ebc:      	movq	%r13, 0x40(%rbx)
 3878ec0:      	movq	%rcx, 0x48(%rbx)
 3878ec4:      	movq	0x18(%rsp), %rax
 3878ec9:      	movq	%rax, (%rbx)
 3878ecc:      	movq	0x20(%rsp), %rax
 3878ed1:      	movq	%rax, 0x8(%rbx)
 3878ed5:      	movq	%r15, 0x50(%rbx)
 3878ed9:      	movq	0x58(%rsp), %rax
 3878ede:      	movq	%rax, 0x58(%rbx)
 3878ee2:      	movq	0x50(%rsp), %rax
 3878ee7:      	movq	%rax, 0x60(%rbx)
 3878eeb:      	movq	0x48(%rsp), %rax
 3878ef0:      	movq	%rax, 0x68(%rbx)
 3878ef4:      	movq	0x40(%rsp), %rax
 3878ef9:      	movq	%rax, 0x70(%rbx)
 3878efd:      	movq	%r12, 0x78(%rbx)
 3878f01:      	movq	0x8(%rsp), %rax
 3878f06:      	movq	%rax, 0x28(%rbx)
 3878f0a:      	movaps	0xa0(%rsp), %xmm0
 3878f12:      	movups	%xmm0, 0x30(%rbx)
 3878f16:      	addq	$0xb8, %rsp
 3878f1d:      	popq	%rbx
 3878f1e:      	popq	%r12
 3878f20:      	popq	%r13
 3878f22:      	popq	%r14
 3878f24:      	popq	%r15
 3878f26:      	popq	%rbp
 3878f27:      	retq
 3878f28:      	leaq	0x18d4fb1(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 3878f2f:      	callq	0x3b2b88b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 3878f34:      	jmp	0x3878c86 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x46>
 3878f39:      	callq	*0x18c8139(%rip)        # 0x5141078 <writev+0x5141078>
 3878f3f:      	movl	%eax, %ecx
 3878f41:      	xorb	$0x1, %cl
 3878f44:      	jmp	0x3878db5 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x175>
 3878f49:      	callq	*0x18c8121(%rip)        # 0x5141070 <writev+0x5141070>
 3878f4f:      	jmp	0x3878e47 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x207>
 3878f54:      	callq	*0x18c811e(%rip)        # 0x5141078 <writev+0x5141078>
 3878f5a:      	testb	%al, %al
 3878f5c:      	jne	0x3878de8 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1a8>
 3878f62:      	movb	$0x1, 0x6c(%r14)
 3878f67:      	jmp	0x3878de8 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1a8>
 3878f6c:      	callq	*0x18c8106(%rip)        # 0x5141078 <writev+0x5141078>
 3878f72:      	testb	%al, %al
 3878f74:      	jne	0x3878e35 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1f5>
 3878f7a:      	movb	$0x1, 0x6c(%r14)
 3878f7f:      	jmp	0x3878e35 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x1f5>
 3878f84:      	movq	%rax, %rbx
 3878f87:      	jmp	0x3878f9b <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x35b>
 3878f89:      	movq	%rax, %rbx
 3878f8c:      	movzbl	0x8(%rsp), %esi
 3878f91:      	movq	0x10(%rsp), %rdi
 3878f96:      	callq	0x384cf20 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCscdodAO9FK5_5alloc6string6StringEECs3pwlnhBXFtN_12memra_server>
 3878f9b:      	movq	0x88(%rsp), %rsi
 3878fa3:      	testq	%rsi, %rsi
 3878fa6:      	je	0x3878fbb <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341+0x37b>
 3878fa8:      	movq	0x90(%rsp), %rdi
 3878fb0:      	movl	$0x1, %edx
 3878fb5:      	callq	*0x18c7d85(%rip)        # 0x5140d40 <writev+0x5140d40>
 3878fbb:      	movq	%rbx, %rdi
 3878fbe:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3878fc3:      	callq	*0x18c7def(%rip)        # 0x5140db8 <writev+0x5140db8>
 3878fc9:      	int3
 3878fca:      	int3
 3878fcb:      	int3
 3878fcc:      	int3
 3878fcd:      	int3
 3878fce:      	int3
 3878fcf:      	int3

000000000388bf00 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends>:
 388bf00:      	pushq	%rbx
 388bf01:      	subq	$0xa0, %rsp
 388bf08:      	movl	$0xea60, %edi           # imm = 0xEA60
 388bf0d:      	callq	0x3875410 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth13with_stall_ms>
 388bf12:      	movq	%rax, %rbx
 388bf15:      	movq	%rax, 0x10(%rsp)
 388bf1a:      	leaq	-0x3500fc1(%rip), %rdi  # 0x38af60 <anon.985887fba90c90e54334d6f2bee737a0.4512.llvm.14468555298774319680+0x20>
 388bf21:      	movl	$0x8, %esi
 388bf26:      	movl	$0x1, %edx
 388bf2b:      	callq	0x3980bc0 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_>
 388bf30:      	addq	$0x10, %rbx
 388bf34:      	leaq	-0x3500fdb(%rip), %rsi  # 0x38af60 <anon.985887fba90c90e54334d6f2bee737a0.4512.llvm.14468555298774319680+0x20>
 388bf3b:      	movl	$0x8, %edx
 388bf40:      	movq	%rbx, %rdi
 388bf43:      	movq	%rax, %rcx
 388bf46:      	callq	0x3875620 <_RNvMs2_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_12WorkerHealth14register_route>
 388bf4b:      	movq	%rax, %rbx
 388bf4e:      	movq	%rax, 0x8(%rsp)
 388bf53:      	movzbl	0x98(%rax), %eax
 388bf5a:      	cmpb	$0x3, %al
 388bf5c:      	je	0x388bfc5 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0xc5>
 388bf5e:      	cmpb	$0x0, %fs:-0x1c90
 388bf67:      	je	0x388bf74 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x74>
 388bf69:      	movq	%fs:-0x1c88, %rax
 388bf72:      	jmp	0x388bfba <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0xba>
 388bf74:      	movl	0x18c1f76(%rip), %eax   # 0x514def0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x10>
 388bf7a:      	testl	%eax, %eax
 388bf7c:      	jne	0x388c26f <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36f>
 388bf82:      	movq	0x18c1f57(%rip), %rax   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 388bf89:      	movl	0x18c1f59(%rip), %ecx   # 0x514dee8 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341+0x8>
 388bf8f:      	movq	%rax, 0x18(%rsp)
 388bf94:      	movl	%ecx, 0x20(%rsp)
 388bf98:      	leaq	0x18(%rsp), %rdi
 388bf9d:      	callq	*0x18b4f15(%rip)        # 0x5140eb8 <writev+0x5140eb8>
 388bfa3:      	imulq	$0x3e8, %rax, %rcx      # imm = 0x3E8
 388bfaa:      	movl	%edx, %eax
 388bfac:      	imulq	$0x431bde83, %rax, %rax # imm = 0x431BDE83
 388bfb3:      	shrq	$0x32, %rax
 388bfb7:      	addq	%rcx, %rax
 388bfba:      	movq	%rax, 0x38(%rbx)
 388bfbe:      	movb	$0x1, 0x98(%rbx)
 388bfc5:      	movq	0x8(%rsp), %rdi
 388bfca:      	addq	$0x10, %rdi
 388bfce:      	callq	0x3878810 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>
 388bfd3:      	movq	0x8(%rsp), %rdi
 388bfd8:      	addq	$0x10, %rdi
 388bfdc:      	callq	0x3878810 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>
 388bfe1:      	movq	0x8(%rsp), %rdi
 388bfe6:      	addq	$0x10, %rdi
 388bfea:      	callq	0x38786a0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>
 388bfef:      	movq	0x8(%rsp), %rdi
 388bff4:      	addq	$0x10, %rdi
 388bff8:      	callq	0x3878970 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth16set_idle_if_free>
 388bffd:      	movq	0x8(%rsp), %rsi
 388c002:      	addq	$0x10, %rsi
 388c006:      	leaq	0x18(%rsp), %rdi
 388c00b:      	callq	0x3878c40 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341>
 388c010:      	cmpb	$0x2, 0x98(%rsp)
 388c018:      	jne	0x388c1c6 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2c6>
 388c01e:      	movq	0x28(%rsp), %rsi
 388c023:      	testq	%rsi, %rsi
 388c026:      	je	0x388c038 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x138>
 388c028:      	movq	0x30(%rsp), %rdi
 388c02d:      	movl	$0x1, %edx
 388c032:      	callq	*0x18b4d08(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c038:      	movq	0x40(%rsp), %rsi
 388c03d:      	cmpq	$-0x1, %rsi
 388c041:      	je	0x388c058 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x158>
 388c043:      	testq	%rsi, %rsi
 388c046:      	je	0x388c058 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x158>
 388c048:      	movq	0x48(%rsp), %rdi
 388c04d:      	movl	$0x1, %edx
 388c052:      	callq	*0x18b4ce8(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c058:      	movq	0x8(%rsp), %rdi
 388c05d:      	addq	$0x10, %rdi
 388c061:      	callq	0x38786a0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>
 388c066:      	movq	0x8(%rsp), %rsi
 388c06b:      	addq	$0x10, %rsi
 388c06f:      	leaq	0x18(%rsp), %rdi
 388c074:      	callq	0x3878c40 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341>
 388c079:      	cmpb	$0x1, 0x98(%rsp)
 388c081:      	jne	0x388c1f3 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2f3>
 388c087:      	movq	0x28(%rsp), %rsi
 388c08c:      	testq	%rsi, %rsi
 388c08f:      	je	0x388c0a1 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x1a1>
 388c091:      	movq	0x30(%rsp), %rdi
 388c096:      	movl	$0x1, %edx
 388c09b:      	callq	*0x18b4c9f(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c0a1:      	movq	0x40(%rsp), %rsi
 388c0a6:      	cmpq	$-0x1, %rsi
 388c0aa:      	je	0x388c0c1 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x1c1>
 388c0ac:      	testq	%rsi, %rsi
 388c0af:      	je	0x388c0c1 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x1c1>
 388c0b1:      	movq	0x48(%rsp), %rdi
 388c0b6:      	movl	$0x1, %edx
 388c0bb:      	callq	*0x18b4c7f(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c0c1:      	movq	0x8(%rsp), %rdi
 388c0c6:      	addq	$0x10, %rdi
 388c0ca:      	callq	0x38786a0 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth11end_request>
 388c0cf:      	movq	0x8(%rsp), %rsi
 388c0d4:      	addq	$0x10, %rsi
 388c0d8:      	leaq	0x18(%rsp), %rdi
 388c0dd:      	callq	0x3878c40 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341>
 388c0e2:      	cmpb	$0x1, 0x98(%rsp)
 388c0ea:      	jne	0x388c215 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x315>
 388c0f0:      	movq	0x28(%rsp), %rsi
 388c0f5:      	testq	%rsi, %rsi
 388c0f8:      	je	0x388c10a <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x20a>
 388c0fa:      	movq	0x30(%rsp), %rdi
 388c0ff:      	movl	$0x1, %edx
 388c104:      	callq	*0x18b4c36(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c10a:      	movq	0x40(%rsp), %rsi
 388c10f:      	cmpq	$-0x1, %rsi
 388c113:      	je	0x388c12a <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x22a>
 388c115:      	testq	%rsi, %rsi
 388c118:      	je	0x388c12a <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x22a>
 388c11a:      	movq	0x48(%rsp), %rdi
 388c11f:      	movl	$0x1, %edx
 388c124:      	callq	*0x18b4c16(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c12a:      	movq	0x8(%rsp), %rdi
 388c12f:      	addq	$0x10, %rdi
 388c133:      	callq	0x3878810 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth13begin_request>
 388c138:      	movq	0x8(%rsp), %rsi
 388c13d:      	addq	$0x10, %rsi
 388c141:      	leaq	0x18(%rsp), %rdi
 388c146:      	callq	0x3878c40 <_RNvMs3_NtCs3pwlnhBXFtN_12memra_server6healthNtB5_11RouteHealth8snapshot.llvm.1341170882515377341>
 388c14b:      	cmpb	$0x2, 0x98(%rsp)
 388c153:      	jne	0x388c242 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x342>
 388c159:      	movq	0x28(%rsp), %rsi
 388c15e:      	testq	%rsi, %rsi
 388c161:      	je	0x388c173 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x273>
 388c163:      	movq	0x30(%rsp), %rdi
 388c168:      	movl	$0x1, %edx
 388c16d:      	callq	*0x18b4bcd(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c173:      	movq	0x40(%rsp), %rsi
 388c178:      	cmpq	$-0x1, %rsi
 388c17c:      	je	0x388c193 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x293>
 388c17e:      	testq	%rsi, %rsi
 388c181:      	je	0x388c193 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x293>
 388c183:      	movq	0x48(%rsp), %rdi
 388c188:      	movl	$0x1, %edx
 388c18d:      	callq	*0x18b4bad(%rip)        # 0x5140d40 <writev+0x5140d40>
 388c193:      	movq	0x8(%rsp), %rax
 388c198:      	lock
 388c199:      	decq	(%rax)
 388c19c:      	jne	0x388c1a8 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2a8>
 388c19e:      	leaq	0x8(%rsp), %rdi
 388c1a3:      	callq	0x39bcf00 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 388c1a8:      	movq	0x10(%rsp), %rax
 388c1ad:      	lock
 388c1ae:      	decq	(%rax)
 388c1b1:      	jne	0x388c1bd <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x2bd>
 388c1b3:      	leaq	0x10(%rsp), %rdi
 388c1b8:      	callq	0x39bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 388c1bd:      	addq	$0xa0, %rsp
 388c1c4:      	popq	%rbx
 388c1c5:      	retq
 388c1c6:      	leaq	0x98(%rsp), %rsi
 388c1ce:      	leaq	-0x2ec86f4(%rip), %rdx  # 0x9c3ae1 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.608.llvm.1341170882515377341+0x1888c>
 388c1d5:      	leaq	-0x2ec7b00(%rip), %rcx  # 0x9c46dc <anon.bd10a02e3ec00da32c2ae70ffdaca96c.779.llvm.1341170882515377341+0x430>
 388c1dc:      	leaq	0x178ad7d(%rip), %r9    # 0x5016f60 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.1341170882515377341+0x3f0>
 388c1e3:      	movl	$0x3d, %r8d
 388c1e9:      	xorl	%edi, %edi
 388c1eb:      	callq	*0x18b6257(%rip)        # 0x5142448 <writev+0x5142448>
 388c1f1:      	jmp	0x388c26d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36d>
 388c1f3:      	leaq	0x98(%rsp), %rsi
 388c1fb:      	leaq	-0x2ee23d0(%rip), %rdx  # 0x9a9e32 <anon.985887fba90c90e54334d6f2bee737a0.8597.llvm.14468555298774319680+0xf40>
 388c202:      	leaq	0x178ad3f(%rip), %r9    # 0x5016f48 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.1341170882515377341+0x3d8>
 388c209:      	xorl	%edi, %edi
 388c20b:      	xorl	%ecx, %ecx
 388c20d:      	callq	*0x18b6235(%rip)        # 0x5142448 <writev+0x5142448>
 388c213:      	jmp	0x388c26d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36d>
 388c215:      	leaq	0x98(%rsp), %rsi
 388c21d:      	leaq	-0x2ee23f2(%rip), %rdx  # 0x9a9e32 <anon.985887fba90c90e54334d6f2bee737a0.8597.llvm.14468555298774319680+0xf40>
 388c224:      	leaq	-0x2ec7b6d(%rip), %rcx  # 0x9c46be <anon.bd10a02e3ec00da32c2ae70ffdaca96c.779.llvm.1341170882515377341+0x412>
 388c22b:      	leaq	0x178acfe(%rip), %r9    # 0x5016f30 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.1341170882515377341+0x3c0>
 388c232:      	movl	$0x3d, %r8d
 388c238:      	xorl	%edi, %edi
 388c23a:      	callq	*0x18b6208(%rip)        # 0x5142448 <writev+0x5142448>
 388c240:      	jmp	0x388c26d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x36d>
 388c242:      	leaq	0x98(%rsp), %rsi
 388c24a:      	leaq	-0x2ec8770(%rip), %rdx  # 0x9c3ae1 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.608.llvm.1341170882515377341+0x1888c>
 388c251:      	leaq	-0x2ec7bb7(%rip), %rcx  # 0x9c46a1 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.779.llvm.1341170882515377341+0x3f5>
 388c258:      	leaq	0x178acb9(%rip), %r9    # 0x5016f18 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.781.llvm.1341170882515377341+0x3a8>
 388c25f:      	movl	$0x3b, %r8d
 388c265:      	xorl	%edi, %edi
 388c267:      	callq	*0x18b61db(%rip)        # 0x5142448 <writev+0x5142448>
 388c26d:      	ud2
 388c26f:      	leaq	0x18c1c6a(%rip), %rdi   # 0x514dee0 <_RNvNvNtCs3pwlnhBXFtN_12memra_server6health5epoch1E.llvm.1341170882515377341>
 388c276:      	callq	0x3b2b88b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>
 388c27b:      	jmp	0x388bf82 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x82>
 388c280:      	movq	%rax, %rbx
 388c283:      	jmp	0x388c2b2 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x3b2>
 388c285:      	movq	%rax, %rbx
 388c288:      	jmp	0x388c29d <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x39d>
 388c28a:      	jmp	0x388c290 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x390>
 388c28c:      	jmp	0x388c290 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x390>
 388c28e:      	jmp	0x388c290 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x390>
 388c290:      	movq	%rax, %rbx
 388c293:      	leaq	0x18(%rsp), %rdi
 388c298:      	callq	0x384d590 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6health13RouteSnapshotEBF_>
 388c29d:      	movq	0x8(%rsp), %rax
 388c2a2:      	lock
 388c2a3:      	decq	(%rax)
 388c2a6:      	jne	0x388c2b2 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x3b2>
 388c2a8:      	leaq	0x8(%rsp), %rdi
 388c2ad:      	callq	0x39bcf00 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 388c2b2:      	movq	0x10(%rsp), %rax
 388c2b7:      	lock
 388c2b8:      	decq	(%rax)
 388c2bb:      	jne	0x388c2c7 <_RNvNtNtCs3pwlnhBXFtN_12memra_server6health5testss_60a_multi_session_route_stays_busy_until_its_last_request_ends+0x3c7>
 388c2bd:      	leaq	0x10(%rsp), %rdi
 388c2c2:      	callq	0x39bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>
 388c2c7:      	movq	%rbx, %rdi
 388c2ca:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 388c2cf:      	callq	*0x18b4ae3(%rip)        # 0x5140db8 <writev+0x5140db8>
 388c2d5:      	int3
 388c2d6:      	int3
 388c2d7:      	int3
 388c2d8:      	int3
 388c2d9:      	int3
 388c2da:      	int3
 388c2db:      	int3
 388c2dc:      	int3
 388c2dd:      	int3
 388c2de:      	int3
 388c2df:      	int3

0000000003980bc0 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_>:
 3980bc0:      	pushq	%r15
 3980bc2:      	pushq	%r14
 3980bc4:      	pushq	%r12
 3980bc6:      	pushq	%rbx
 3980bc7:      	subq	$0x2c8, %rsp            # imm = 0x2C8
 3980bce:      	movq	%rdx, %r15
 3980bd1:      	movq	%rsi, %rbx
 3980bd4:      	testq	%rsi, %rsi
 3980bd7:      	je	0x3980c0d <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x4d>
 3980bd9:      	movq	%rdi, %r12
 3980bdc:      	callq	*0x17c01ae(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980be2:      	movl	$0x1, %esi
 3980be7:      	movq	%rbx, %rdi
 3980bea:      	callq	*0x17c01a8(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980bf0:      	testq	%rax, %rax
 3980bf3:      	je	0x3980e8b <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2cb>
 3980bf9:      	movq	%rax, %r14
 3980bfc:      	movq	%rax, %rdi
 3980bff:      	movq	%r12, %rsi
 3980c02:      	movq	%rbx, %rdx
 3980c05:      	callq	*0x17c012d(%rip)        # 0x5140d38 <writev+0x5140d38>
 3980c0b:      	jmp	0x3980c13 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x53>
 3980c0d:      	movl	$0x1, %r14d
 3980c13:      	cmpq	$0x1, %r15
 3980c17:      	adcq	$0x0, %r15
 3980c1b:      	callq	*0x17c016f(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980c21:      	movl	$0x200, %edi            # imm = 0x200
 3980c26:      	movl	$0x8, %esi
 3980c2b:      	callq	*0x17c0167(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980c31:      	testq	%rax, %rax
 3980c34:      	je	0x3980e67 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2a7>
 3980c3a:      	movq	%rax, %r12
 3980c3d:      	callq	*0x17c014d(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980c43:      	movl	$0x800, %edi            # imm = 0x800
 3980c48:      	movl	$0x8, %esi
 3980c4d:      	callq	*0x17c0145(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980c53:      	testq	%rax, %rax
 3980c56:      	je	0x3980e79 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2b9>
 3980c5c:      	xorps	%xmm0, %xmm0
 3980c5f:      	movups	%xmm0, 0x100(%rsp)
 3980c67:      	movups	%xmm0, 0x118(%rsp)
 3980c6f:      	movups	%xmm0, 0x128(%rsp)
 3980c77:      	movups	%xmm0, 0x138(%rsp)
 3980c7f:      	movups	%xmm0, 0x148(%rsp)
 3980c87:      	movups	%xmm0, 0x158(%rsp)
 3980c8f:      	movups	%xmm0, 0x168(%rsp)
 3980c97:      	movups	%xmm0, 0x178(%rsp)
 3980c9f:      	movups	%xmm0, 0x188(%rsp)
 3980ca7:      	movups	%xmm0, 0x198(%rsp)
 3980caf:      	movups	%xmm0, 0x1a8(%rsp)
 3980cb7:      	movups	%xmm0, 0x1b8(%rsp)
 3980cbf:      	movups	%xmm0, 0x1c8(%rsp)
 3980cc7:      	movups	%xmm0, 0x1d8(%rsp)
 3980ccf:      	movups	%xmm0, 0x1e8(%rsp)
 3980cd7:      	movups	%xmm0, 0x1f8(%rsp)
 3980cdf:      	movups	%xmm0, 0x208(%rsp)
 3980ce7:      	movups	%xmm0, 0x218(%rsp)
 3980cef:      	movups	%xmm0, 0x228(%rsp)
 3980cf7:      	movups	%xmm0, 0x2b8(%rsp)
 3980cff:      	movups	%xmm0, 0x2a8(%rsp)
 3980d07:      	movups	%xmm0, 0x298(%rsp)
 3980d0f:      	movups	%xmm0, 0x288(%rsp)
 3980d17:      	movups	%xmm0, 0x278(%rsp)
 3980d1f:      	movups	%xmm0, 0x268(%rsp)
 3980d27:      	movups	%xmm0, 0x258(%rsp)
 3980d2f:      	movups	%xmm0, 0x248(%rsp)
 3980d37:      	movups	%xmm0, 0x238(%rsp)
 3980d3f:      	movq	%rbx, 0x10(%rsp)
 3980d44:      	movq	%r14, 0x18(%rsp)
 3980d49:      	movq	%rbx, 0x20(%rsp)
 3980d4e:      	movl	$0x0, 0xb8(%rsp)
 3980d59:      	movb	$0x0, 0xbc(%rsp)
 3980d61:      	movups	%xmm0, 0x30(%rsp)
 3980d66:      	movups	%xmm0, 0x40(%rsp)
 3980d6b:      	movups	%xmm0, 0x50(%rsp)
 3980d70:      	movq	$0x40, 0xc0(%rsp)
 3980d7c:      	movq	%r12, 0xc8(%rsp)
 3980d84:      	movups	%xmm0, 0xd0(%rsp)
 3980d8c:      	movq	$0x40, 0xe0(%rsp)
 3980d98:      	movl	$0x0, 0xe8(%rsp)
 3980da3:      	movb	$0x0, 0xec(%rsp)
 3980dab:      	movq	$0x100, 0xf0(%rsp)      # imm = 0x100
 3980db7:      	movq	%rax, 0xf8(%rsp)
 3980dbf:      	movq	$0x100, 0x110(%rsp)     # imm = 0x100
 3980dcb:      	movq	$0x1, (%rsp)
 3980dd3:      	movq	$0x1, 0x8(%rsp)
 3980ddc:      	movq	%r15, 0x28(%rsp)
 3980de1:      	movups	%xmm0, 0x60(%rsp)
 3980de6:      	movups	%xmm0, 0x70(%rsp)
 3980deb:      	movups	%xmm0, 0x80(%rsp)
 3980df3:      	movups	%xmm0, 0x90(%rsp)
 3980dfb:      	movups	%xmm0, 0xa0(%rsp)
 3980e03:      	movq	$0x0, 0xb0(%rsp)
 3980e0f:      	callq	*0x17bff7b(%rip)        # 0x5140d90 <writev+0x5140d90>
 3980e15:      	movl	$0x2c8, %edi            # imm = 0x2C8
 3980e1a:      	movl	$0x8, %esi
 3980e1f:      	callq	*0x17bff73(%rip)        # 0x5140d98 <writev+0x5140d98>
 3980e25:      	testq	%rax, %rax
 3980e28:      	je	0x3980e50 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x290>
 3980e2a:      	movq	%rsp, %rsi
 3980e2d:      	movl	$0x2c8, %edx            # imm = 0x2C8
 3980e32:      	movq	%rax, %rdi
 3980e35:      	movq	%rax, %rbx
 3980e38:      	callq	*0x17bfefa(%rip)        # 0x5140d38 <writev+0x5140d38>
 3980e3e:      	movq	%rbx, %rax
 3980e41:      	addq	$0x2c8, %rsp            # imm = 0x2C8
 3980e48:      	popq	%rbx
 3980e49:      	popq	%r12
 3980e4b:      	popq	%r14
 3980e4d:      	popq	%r15
 3980e4f:      	retq
 3980e50:      	leaq	0x10(%rsp), %rbx
 3980e55:      	movl	$0x8, %edi
 3980e5a:      	movl	$0x2c8, %esi            # imm = 0x2C8
 3980e5f:      	callq	*0x17c00fb(%rip)        # 0x5140f60 <writev+0x5140f60>
 3980e65:      	jmp	0x3980e89 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2c9>
 3980e67:      	movl	$0x8, %edi
 3980e6c:      	movl	$0x200, %esi            # imm = 0x200
 3980e71:      	callq	*0x17bff11(%rip)        # 0x5140d88 <writev+0x5140d88>
 3980e77:      	jmp	0x3980e89 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2c9>
 3980e79:      	movl	$0x8, %edi
 3980e7e:      	movl	$0x800, %esi            # imm = 0x800
 3980e83:      	callq	*0x17bfeff(%rip)        # 0x5140d88 <writev+0x5140d88>
 3980e89:      	ud2
 3980e8b:      	movl	$0x1, %edi
 3980e90:      	movq	%rbx, %rsi
 3980e93:      	callq	*0x17bfeef(%rip)        # 0x5140d88 <writev+0x5140d88>
 3980e99:      	movq	%rax, %r15
 3980e9c:      	movl	$0x200, %esi            # imm = 0x200
 3980ea1:      	movl	$0x8, %edx
 3980ea6:      	movq	%r12, %rdi
 3980ea9:      	callq	*0x17bfe91(%rip)        # 0x5140d40 <writev+0x5140d40>
 3980eaf:      	jmp	0x3980eb4 <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x2f4>
 3980eb1:      	movq	%rax, %r15
 3980eb4:      	testq	%rbx, %rbx
 3980eb7:      	je	0x3980edd <_RINvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB5_9RouteLoad3newReEB7_+0x31d>
 3980eb9:      	movl	$0x1, %edx
 3980ebe:      	movq	%r14, %rdi
 3980ec1:      	movq	%rbx, %rsi
 3980ec4:      	callq	*0x17bfe76(%rip)        # 0x5140d40 <writev+0x5140d40>
 3980eca:      	movq	%r15, %rdi
 3980ecd:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3980ed2:      	movq	%rax, %r15
 3980ed5:      	movq	%rbx, %rdi
 3980ed8:      	callq	0x398c5e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadEBF_>
 3980edd:      	movq	%r15, %rdi
 3980ee0:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3980ee5:      	int3
 3980ee6:      	int3
 3980ee7:      	int3
 3980ee8:      	int3
 3980ee9:      	int3
 3980eea:      	int3
 3980eeb:      	int3
 3980eec:      	int3
 3980eed:      	int3
 3980eee:      	int3
 3980eef:      	int3

00000000039bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>:
 39bcc70:      	pushq	%rbx
 39bcc71:      	movq	(%rdi), %rbx
 39bcc74:      	movq	0x10(%rbx), %rsi
 39bcc78:      	testq	%rsi, %rsi
 39bcc7b:      	je	0x39bcc8c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x1c>
 39bcc7d:      	movq	0x18(%rbx), %rdi
 39bcc81:      	movl	$0x1, %edx
 39bcc86:      	callq	*0x17840b4(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcc8c:      	movq	0xc0(%rbx), %rsi
 39bcc93:      	testq	%rsi, %rsi
 39bcc96:      	je	0x39bccae <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x3e>
 39bcc98:      	movq	0xc8(%rbx), %rdi
 39bcc9f:      	shlq	$0x3, %rsi
 39bcca3:      	movl	$0x8, %edx
 39bcca8:      	callq	*0x1784092(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bccae:      	movq	0xf0(%rbx), %rsi
 39bccb5:      	testq	%rsi, %rsi
 39bccb8:      	je	0x39bccd0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x60>
 39bccba:      	movq	0xf8(%rbx), %rdi
 39bccc1:      	shlq	$0x3, %rsi
 39bccc5:      	movl	$0x8, %edx
 39bccca:      	callq	*0x1784070(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bccd0:      	cmpq	$-0x1, %rbx
 39bccd4:      	je	0x39bccf1 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x81>
 39bccd6:      	lock
 39bccd7:      	decq	0x8(%rbx)
 39bccdb:      	jne	0x39bccf1 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_+0x81>
 39bccdd:      	movl	$0x2c8, %esi            # imm = 0x2C8
 39bcce2:      	movl	$0x8, %edx
 39bcce7:      	movq	%rbx, %rdi
 39bccea:      	popq	%rbx
 39bcceb:      	jmpq	*0x178404f(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bccf1:      	popq	%rbx
 39bccf2:      	retq
 39bccf3:      	int3
 39bccf4:      	int3
 39bccf5:      	int3
 39bccf6:      	int3
 39bccf7:      	int3
 39bccf8:      	int3
 39bccf9:      	int3
 39bccfa:      	int3
 39bccfb:      	int3
 39bccfc:      	int3
 39bccfd:      	int3
 39bccfe:      	int3
 39bccff:      	int3

00000000039bcf00 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>:
 39bcf00:      	pushq	%rbx
 39bcf01:      	movq	(%rdi), %rbx
 39bcf04:      	movq	0x10(%rbx), %rsi
 39bcf08:      	testq	%rsi, %rsi
 39bcf0b:      	je	0x39bcf1c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x1c>
 39bcf0d:      	movq	0x18(%rbx), %rdi
 39bcf11:      	movl	$0x1, %edx
 39bcf16:      	callq	*0x1783e24(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf1c:      	movq	0x80(%rbx), %rsi
 39bcf23:      	testq	%rsi, %rsi
 39bcf26:      	je	0x39bcf3a <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x3a>
 39bcf28:      	movq	0x88(%rbx), %rdi
 39bcf2f:      	movl	$0x1, %edx
 39bcf34:      	callq	*0x1783e06(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf3a:      	movq	0x28(%rbx), %rax
 39bcf3e:      	lock
 39bcf3f:      	decq	(%rax)
 39bcf42:      	jne	0x39bcf4d <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x4d>
 39bcf44:      	leaq	0x28(%rbx), %rdi
 39bcf48:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 39bcf4d:      	cmpq	$-0x1, %rbx
 39bcf51:      	je	0x39bcf6e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x6e>
 39bcf53:      	lock
 39bcf54:      	decq	0x8(%rbx)
 39bcf58:      	jne	0x39bcf6e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_+0x6e>
 39bcf5a:      	movl	$0xa0, %esi
 39bcf5f:      	movl	$0x8, %edx
 39bcf64:      	movq	%rbx, %rdi
 39bcf67:      	popq	%rbx
 39bcf68:      	jmpq	*0x1783dd2(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf6e:      	popq	%rbx
 39bcf6f:      	retq

00000000039bcf70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_>:
 39bcf70:      	pushq	%r15
 39bcf72:      	pushq	%r14
 39bcf74:      	pushq	%rbx
 39bcf75:      	movq	(%rdi), %rbx
 39bcf78:      	movq	0x18(%rbx), %rsi
 39bcf7c:      	testq	%rsi, %rsi
 39bcf7f:      	je	0x39bcf90 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x20>
 39bcf81:      	movq	0x20(%rbx), %rdi
 39bcf85:      	movl	$0x1, %edx
 39bcf8a:      	callq	*0x1783db0(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcf90:      	movq	0x38(%rbx), %rsi
 39bcf94:      	testq	%rsi, %rsi
 39bcf97:      	je	0x39bcfa8 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x38>
 39bcf99:      	movq	0x40(%rbx), %rdi
 39bcf9d:      	movl	$0x1, %edx
 39bcfa2:      	callq	*0x1783d98(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcfa8:      	movq	0x58(%rbx), %rsi
 39bcfac:      	testq	%rsi, %rsi
 39bcfaf:      	je	0x39bcfc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x50>
 39bcfb1:      	movq	0x60(%rbx), %rdi
 39bcfb5:      	movl	$0x1, %edx
 39bcfba:      	callq	*0x1783d80(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bcfc0:      	movq	0x70(%rbx), %rax
 39bcfc4:      	testq	%rax, %rax
 39bcfc7:      	je	0x39bcfd8 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x68>
 39bcfc9:      	lock
 39bcfca:      	decq	(%rax)
 39bcfcd:      	jne	0x39bcfd8 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x68>
 39bcfcf:      	leaq	0x70(%rbx), %rdi
 39bcfd3:      	callq	0x39b9fc0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcDNtNtCs3pwlnhBXFtN_12memra_server8metering8MeteringEL_E9drop_slowBK_>
 39bcfd8:      	movq	0xd8(%rbx), %r15
 39bcfdf:      	testq	%r15, %r15
 39bcfe2:      	je	0x39bd00c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x9c>
 39bcfe4:      	movq	0xd0(%rbx), %r14
 39bcfeb:      	jmp	0x39bcff9 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x89>
 39bcfed:      	nopl	(%rax)
 39bcff0:      	addq	$0x8, %r14
 39bcff4:      	decq	%r15
 39bcff7:      	je	0x39bd00c <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x9c>
 39bcff9:      	movq	(%r14), %rax
 39bcffc:      	lock
 39bcffd:      	decq	(%rax)
 39bd000:      	jne	0x39bcff0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x80>
 39bd002:      	movq	%r14, %rdi
 39bd005:      	callq	0x39bcf00 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthE9drop_slowBJ_>
 39bd00a:      	jmp	0x39bcff0 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x80>
 39bd00c:      	movq	0xc8(%rbx), %rsi
 39bd013:      	testq	%rsi, %rsi
 39bd016:      	je	0x39bd02e <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0xbe>
 39bd018:      	movq	0xd0(%rbx), %rdi
 39bd01f:      	shlq	$0x3, %rsi
 39bd023:      	movl	$0x8, %edx
 39bd028:      	callq	*0x1783d12(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bd02e:      	cmpq	$-0x1, %rbx
 39bd032:      	je	0x39bd053 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0xe3>
 39bd034:      	lock
 39bd035:      	decq	0x8(%rbx)
 39bd039:      	jne	0x39bd053 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0xe3>
 39bd03b:      	movl	$0xe8, %esi
 39bd040:      	movl	$0x8, %edx
 39bd045:      	movq	%rbx, %rdi
 39bd048:      	popq	%rbx
 39bd049:      	popq	%r14
 39bd04b:      	popq	%r15
 39bd04d:      	jmpq	*0x1783ced(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bd053:      	popq	%rbx
 39bd054:      	popq	%r14
 39bd056:      	popq	%r15
 39bd058:      	retq
 39bd059:      	movq	%rax, %r14
 39bd05c:      	leaq	0xb8(%rbx), %rdi
 39bd063:      	callq	0x3989590 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison6rwlock6RwLockINtNtCscdodAO9FK5_5alloc3vec3VecINtNtB1x_4sync3ArcNtNtCs3pwlnhBXFtN_12memra_server6health11RouteHealthEEEEB2k_>
 39bd068:      	cmpq	$-0x1, %rbx
 39bd06c:      	je	0x39bd088 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x118>
 39bd06e:      	lock
 39bd06f:      	decq	0x8(%rbx)
 39bd073:      	jne	0x39bd088 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server6health12WorkerHealthE9drop_slowBJ_+0x118>
 39bd075:      	movl	$0xe8, %esi
 39bd07a:      	movl	$0x8, %edx
 39bd07f:      	movq	%rbx, %rdi
 39bd082:      	callq	*0x1783cb8(%rip)        # 0x5140d40 <writev+0x5140d40>
 39bd088:      	movq	%r14, %rdi
 39bd08b:      	callq	0x4fde6d0 <_Unwind_Resume@plt>

0000000003b1a510 <_RNvXs_NtNtCs2AWtUsOyxgP_3std4sync6poisonINtB4_11PoisonErrorINtNtB4_5mutex10MutexGuardINtNtCscdodAO9FK5_5alloc3vec3VecNtNtCs3pwlnhBXFtN_12memra_server5tests10MeterEventEEENtNtCs4NRVxsYgnAr_4core3fmt5Debug3fmtB1V_>:
 3b1a510:      	pushq	%rbx
 3b1a511:      	subq	$0x10, %rsp
 3b1a515:      	leaq	-0x2d01210(%rip), %rdx  # 0xe1930c <anon.0cae78d1abade22e61931222b273b913.50.llvm.7562590961585166920+0x396>
 3b1a51c:      	movq	%rsp, %rbx
 3b1a51f:      	movl	$0xb, %ecx
 3b1a524:      	movq	%rbx, %rdi
 3b1a527:      	callq	*0x1628893(%rip)        # 0x5142dc0 <writev+0x5142dc0>
 3b1a52d:      	movq	%rbx, %rdi
 3b1a530:      	callq	*0x1628dea(%rip)        # 0x5143320 <writev+0x5143320>
 3b1a536:      	addq	$0x10, %rsp
 3b1a53a:      	popq	%rbx
 3b1a53b:      	retq
 3b1a53c:      	int3
 3b1a53d:      	int3
 3b1a53e:      	int3
 3b1a53f:      	int3

0000000003b2b258 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>:
 3b2b258:      	movl	0x18(%rdi), %eax
 3b2b25b:      	testl	%eax, %eax
 3b2b25d:      	jne	0x3b2b260 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_+0x8>
 3b2b25f:      	retq
 3b2b260:      	subq	$0x28, %rsp
 3b2b264:      	leaq	0x18(%rsp), %rax
 3b2b269:      	movq	%rdi, (%rax)
 3b2b26c:      	addq	$0x18, %rdi
 3b2b270:      	leaq	0xf(%rsp), %rcx
 3b2b275:      	movq	%rcx, 0x8(%rax)
 3b2b279:      	leaq	0x10(%rsp), %rdx
 3b2b27e:      	movq	%rax, (%rdx)
 3b2b281:      	leaq	0x14fba40(%rip), %rcx   # 0x5026cc8 <anon.3222e37afe7119ca9a4f598626aaaddd.38.llvm.15726592429269621555>
 3b2b288:      	leaq	0x14fba61(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2b28f:      	pushq	$0x1
 3b2b291:      	popq	%rsi
 3b2b292:      	callq	*0x1615b08(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2b298:      	addq	$0x28, %rsp
 3b2b29c:      	retq

0000000003b2b3b1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>:
 3b2b3b1:      	movl	0x10(%rdi), %eax
 3b2b3b4:      	testl	%eax, %eax
 3b2b3b6:      	jne	0x3b2b3b9 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_+0x8>
 3b2b3b8:      	retq
 3b2b3b9:      	subq	$0x28, %rsp
 3b2b3bd:      	leaq	0x18(%rsp), %rax
 3b2b3c2:      	movq	%rdi, (%rax)
 3b2b3c5:      	addq	$0x10, %rdi
 3b2b3c9:      	leaq	0xf(%rsp), %rcx
 3b2b3ce:      	movq	%rcx, 0x8(%rax)
 3b2b3d2:      	leaq	0x10(%rsp), %rdx
 3b2b3d7:      	movq	%rax, (%rdx)
 3b2b3da:      	leaq	0x14fb9c7(%rip), %rcx   # 0x5026da8 <anon.3222e37afe7119ca9a4f598626aaaddd.46.llvm.15726592429269621555>
 3b2b3e1:      	leaq	0x14fb908(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2b3e8:      	pushq	$0x1
 3b2b3ea:      	popq	%rsi
 3b2b3eb:      	callq	*0x16159af(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2b3f1:      	addq	$0x28, %rsp
 3b2b3f5:      	retq

0000000003b2b88b <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server>:
 3b2b88b:      	movl	0x10(%rdi), %eax
 3b2b88e:      	testl	%eax, %eax
 3b2b890:      	jne	0x3b2b893 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockNtNtB7_4time7InstantE10initializeNCINvB2_11get_or_initNvMBV_BT_3nowE0zECs3pwlnhBXFtN_12memra_server+0x8>
 3b2b892:      	retq
 3b2b893:      	subq	$0x28, %rsp
 3b2b897:      	leaq	0x18(%rsp), %rax
 3b2b89c:      	movq	%rdi, (%rax)
 3b2b89f:      	addq	$0x10, %rdi
 3b2b8a3:      	leaq	0xf(%rsp), %rcx
 3b2b8a8:      	movq	%rcx, 0x8(%rax)
 3b2b8ac:      	leaq	0x10(%rsp), %rdx
 3b2b8b1:      	movq	%rax, (%rdx)
 3b2b8b4:      	leaq	0x14fb7bd(%rip), %rcx   # 0x5027078 <anon.3222e37afe7119ca9a4f598626aaaddd.65.llvm.15726592429269621555>
 3b2b8bb:      	leaq	0x14fb42e(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2b8c2:      	pushq	$0x1
 3b2b8c4:      	popq	%rsi
 3b2b8c5:      	callq	*0x16154d5(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2b8cb:      	addq	$0x28, %rsp
 3b2b8cf:      	retq

0000000003b2c977 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_>:
 3b2c977:      	movl	0x8(%rdi), %eax
 3b2c97a:      	testl	%eax, %eax
 3b2c97c:      	jne	0x3b2c97f <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zEB1w_+0x8>
 3b2c97e:      	retq
 3b2c97f:      	subq	$0x28, %rsp
 3b2c983:      	leaq	0x18(%rsp), %rax
 3b2c988:      	movq	%rdi, (%rax)
 3b2c98b:      	addq	$0x8, %rdi
 3b2c98f:      	leaq	0xf(%rsp), %rcx
 3b2c994:      	movq	%rcx, 0x8(%rax)
 3b2c998:      	leaq	0x10(%rsp), %rdx
 3b2c99d:      	movq	%rax, (%rdx)
 3b2c9a0:      	leaq	0x14fb0a9(%rip), %rcx   # 0x5027a50 <anon.3222e37afe7119ca9a4f598626aaaddd.130.llvm.15726592429269621555>
 3b2c9a7:      	leaq	0x14fa342(%rip), %r8    # 0x5026cf0 <anon.3222e37afe7119ca9a4f598626aaaddd.40.llvm.15726592429269621555>
 3b2c9ae:      	pushq	$0x1
 3b2c9b0:      	popq	%rsi
 3b2c9b1:      	callq	*0x16143e9(%rip)        # 0x5140da0 <writev+0x5140da0>
 3b2c9b7:      	addq	$0x28, %rsp
 3b2c9bb:      	retq

0000000003b49e20 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 3b49e20:      	pushq	%r14
 3b49e22:      	pushq	%rbx
 3b49e23:      	pushq	%rax
 3b49e24:      	movq	%rdi, %rbx
 3b49e27:      	cmpb	$0x1, 0x31(%rdi)
 3b49e2b:      	jne	0x3b49e81 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3b49e2d:      	movq	0x8(%rbx), %rcx
 3b49e31:      	movq	(%rcx), %rax
 3b49e34:      	nopw	%cs:(%rax,%rax)
 3b49e40:      	testq	%rax, %rax
 3b49e43:      	je	0x3b49e50 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 3b49e45:      	leaq	-0x1(%rax), %rdx
 3b49e49:      	lock
 3b49e4a:      	cmpxchgq	%rdx, (%rcx)
 3b49e4e:      	jne	0x3b49e40 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 3b49e50:      	cmpb	$0x0, 0x32(%rbx)
 3b49e54:      	jne	0x3b49e81 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3b49e56:      	movq	(%rbx), %rcx
 3b49e59:      	movzbl	0x30(%rbx), %edx
 3b49e5d:      	movq	(%rcx,%rdx,8), %rax
 3b49e61:      	nopw	%cs:(%rax,%rax)
 3b49e70:      	testq	%rax, %rax
 3b49e73:      	je	0x3b49e81 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3b49e75:      	leaq	-0x1(%rax), %rsi
 3b49e79:      	lock
 3b49e7a:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 3b49e7f:      	jne	0x3b49e70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 3b49e81:      	cmpl	$-0x1, 0x28(%rbx)
 3b49e85:      	je	0x3b49eca <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3b49e87:      	movq	0x10(%rbx), %rdi
 3b49e8b:      	cmpq	$0x2, %rdi
 3b49e8f:      	ja	0x3b49ed2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 3b49e91:      	movq	0x18(%rbx), %rcx
 3b49e95:      	addq	$0x18, %rbx
 3b49e99:      	movq	0x30(%rcx,%rdi,8), %rax
 3b49e9e:      	nop
 3b49ea0:      	testq	%rax, %rax
 3b49ea3:      	je	0x3b49eb2 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 3b49ea5:      	leaq	-0x1(%rax), %rdx
 3b49ea9:      	lock
 3b49eaa:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 3b49eb0:      	jne	0x3b49ea0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 3b49eb2:      	movq	(%rbx), %rax
 3b49eb5:      	lock
 3b49eb6:      	decq	(%rax)
 3b49eb9:      	jne	0x3b49eca <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3b49ebb:      	movq	%rbx, %rdi
 3b49ebe:      	addq	$0x8, %rsp
 3b49ec2:      	popq	%rbx
 3b49ec3:      	popq	%r14
 3b49ec5:      	jmp	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3b49eca:      	addq	$0x8, %rsp
 3b49ece:      	popq	%rbx
 3b49ecf:      	popq	%r14
 3b49ed1:      	retq
 3b49ed2:      	leaq	0x14d7ff7(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 3b49ed9:      	movl	$0x3, %esi
 3b49ede:      	callq	*0x15f6f94(%rip)        # 0x5140e78 <writev+0x5140e78>
 3b49ee4:      	ud2
 3b49ee6:      	movq	%rax, %r14
 3b49ee9:      	movq	0x18(%rbx), %rax
 3b49eed:      	lock
 3b49eee:      	decq	(%rax)
 3b49ef1:      	jne	0x3b49eff <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 3b49ef3:      	addq	$0x18, %rbx
 3b49ef7:      	movq	%rbx, %rdi
 3b49efa:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3b49eff:      	movq	%r14, %rdi
 3b49f02:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3b49f07:      	callq	*0x15f6eab(%rip)        # 0x5140db8 <writev+0x5140db8>
 3b49f0d:      	int3
 3b49f0e:      	int3
 3b49f0f:      	int3

0000000003b55f60 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockAjj3_E10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0B2o_.llvm.15726592429269621555>:
 3b55f60:      	pushq	%r14
 3b55f62:      	pushq	%rbx
 3b55f63:      	subq	$0x48, %rsp
 3b55f67:      	movq	(%rdi), %rax
 3b55f6a:      	movq	(%rax), %r14
 3b55f6d:      	movq	$0x0, (%rax)
 3b55f74:      	testq	%r14, %r14
 3b55f77:      	je	0x3b55f9f <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockAjj3_E10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0B2o_.llvm.15726592429269621555+0x3f>
 3b55f79:      	callq	0x387b890 <_RNvNtCs3pwlnhBXFtN_12memra_server14route_contract22hybrid_interactive_cap>
 3b55f7e:      	movq	%rax, %rbx
 3b55f81:      	movq	%rsp, %rdi
 3b55f84:      	callq	*0x15ebc66(%rip)        # 0x5141bf0 <writev+0x5141bf0>
 3b55f8a:      	movups	0x28(%rsp), %xmm0
 3b55f8f:      	movq	%rbx, (%r14)
 3b55f92:      	movups	%xmm0, 0x8(%r14)
 3b55f97:      	addq	$0x48, %rsp
 3b55f9b:      	popq	%rbx
 3b55f9c:      	popq	%r14
 3b55f9e:      	retq
 3b55f9f:      	leaq	0x14d1faa(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b55fa6:      	callq	*0x15eae94(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b55fac:      	int3
 3b55fad:      	int3
 3b55fae:      	int3
 3b55faf:      	int3

0000000003b56370 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555>:
 3b56370:      	pushq	%r15
 3b56372:      	pushq	%r14
 3b56374:      	pushq	%rbx
 3b56375:      	subq	$0x20, %rsp
 3b56379:      	movq	(%rdi), %rax
 3b5637c:      	movq	(%rax), %r14
 3b5637f:      	movq	$0x0, (%rax)
 3b56386:      	testq	%r14, %r14
 3b56389:      	je	0x3b564ff <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x18f>
 3b5638f:      	leaq	-0x2d3c479(%rip), %rsi  # 0xe19f1d <anon.3222e37afe7119ca9a4f598626aaaddd.276.llvm.15726592429269621555+0xae>
 3b56396:      	movq	%rsp, %rdi
 3b56399:      	movl	$0x15, %edx
 3b5639e:      	callq	*0x15eaaa4(%rip)        # 0x5140e48 <writev+0x5140e48>
 3b563a4:      	cmpl	$0x1, (%rsp)
 3b563a8:      	jne	0x3b563cc <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x5c>
 3b563aa:      	movq	0x8(%rsp), %rsi
 3b563af:      	cmpq	$-0x1, %rsi
 3b563b3:      	je	0x3b563fe <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563b5:      	testq	%rsi, %rsi
 3b563b8:      	je	0x3b563fe <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563ba:      	movq	0x10(%rsp), %rdi
 3b563bf:      	movl	$0x1, %edx
 3b563c4:      	callq	*0x15ea976(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b563ca:      	jmp	0x3b563fe <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563cc:      	movq	0x8(%rsp), %rsi
 3b563d1:      	cmpq	$-0x1, %rsi
 3b563d5:      	je	0x3b563fe <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x8e>
 3b563d7:      	movq	0x10(%rsp), %rdi
 3b563dc:      	movq	0x18(%rsp), %rcx
 3b563e1:      	testq	%rcx, %rcx
 3b563e4:      	je	0x3b56403 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x93>
 3b563e6:      	cmpq	$0x1, %rcx
 3b563ea:      	jne	0x3b56427 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xb7>
 3b563ec:      	movzbl	(%rdi), %eax
 3b563ef:      	xorl	%r15d, %r15d
 3b563f2:      	cmpl	$0x2b, %eax
 3b563f5:      	je	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b563f7:      	cmpl	$0x2d, %eax
 3b563fa:      	je	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b563fc:      	jmp	0x3b5642a <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xba>
 3b563fe:      	xorl	%r15d, %r15d
 3b56401:      	jmp	0x3b56416 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xa6>
 3b56403:      	movq	%rcx, %r15
 3b56406:      	testq	%rsi, %rsi
 3b56409:      	je	0x3b56416 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0xa6>
 3b5640b:      	movl	$0x1, %edx
 3b56410:      	callq	*0x15ea92a(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b56416:      	movq	%r15, (%r14)
 3b56419:      	movq	%rbx, 0x8(%r14)
 3b5641d:      	addq	$0x20, %rsp
 3b56421:      	popq	%rbx
 3b56422:      	popq	%r14
 3b56424:      	popq	%r15
 3b56426:      	retq
 3b56427:      	movzbl	(%rdi), %eax
 3b5642a:      	xorl	%r8d, %r8d
 3b5642d:      	cmpb	$0x2b, %al
 3b5642f:      	sete	%r8b
 3b56433:      	movq	%r8, %rax
 3b56436:      	negq	%rax
 3b56439:      	movq	%rcx, %rdx
 3b5643c:      	subq	%r8, %rdx
 3b5643f:      	addq	%rdi, %r8
 3b56442:      	cmpq	$0x11, %rdx
 3b56446:      	jae	0x3b56497 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x127>
 3b56448:      	movl	$0x1, %r15d
 3b5644e:      	testq	%rdx, %rdx
 3b56451:      	je	0x3b564e5 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x175>
 3b56457:      	addq	%rax, %rcx
 3b5645a:      	negq	%rcx
 3b5645d:      	xorl	%eax, %eax
 3b5645f:      	xorl	%ebx, %ebx
 3b56461:      	nopw	%cs:(%rax,%rax)
 3b56470:      	movzbl	(%r8,%rax), %edx
 3b56475:      	addl	$-0x30, %edx
 3b56478:      	cmpl	$0x9, %edx
 3b5647b:      	ja	0x3b564ec <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x17c>
 3b5647d:      	leaq	(%rbx,%rbx,4), %r9
 3b56481:      	movl	%edx, %edx
 3b56483:      	leaq	(%rdx,%r9,2), %rbx
 3b56487:      	incq	%rax
 3b5648a:      	movq	%rcx, %rdx
 3b5648d:      	addq	%rax, %rdx
 3b56490:      	jne	0x3b56470 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x100>
 3b56492:      	jmp	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b56497:      	addq	%rax, %rcx
 3b5649a:      	negq	%rcx
 3b5649d:      	xorl	%r15d, %r15d
 3b564a0:      	movl	$0xa, %r9d
 3b564a6:      	xorl	%r10d, %r10d
 3b564a9:      	xorl	%ebx, %ebx
 3b564ab:      	movq	%rcx, %rax
 3b564ae:      	addq	%r10, %rax
 3b564b1:      	je	0x3b564f4 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x184>
 3b564b3:      	movq	%rbx, %rax
 3b564b6:      	mulq	%r9
 3b564b9:      	jo	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564bf:      	movq	%rax, %rbx
 3b564c2:      	movzbl	(%r8,%r10), %edx
 3b564c7:      	addl	$-0x30, %edx
 3b564ca:      	addq	%rdx, %rbx
 3b564cd:      	setb	%al
 3b564d0:      	cmpl	$0x9, %edx
 3b564d3:      	ja	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564d9:      	incq	%r10
 3b564dc:      	testb	%al, %al
 3b564de:      	je	0x3b564ab <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x13b>
 3b564e0:      	jmp	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564e5:      	xorl	%ebx, %ebx
 3b564e7:      	jmp	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564ec:      	xorl	%r15d, %r15d
 3b564ef:      	jmp	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564f4:      	movl	$0x1, %r15d
 3b564fa:      	jmp	0x3b56406 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555+0x96>
 3b564ff:      	leaq	0x14d1a4a(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b56506:      	callq	*0x15ea934(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b5650c:      	int3
 3b5650d:      	int3
 3b5650e:      	int3
 3b5650f:      	int3

0000000003b58590 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockNtNtBc_4time7InstantE10initializeNCINvB1a_11get_or_initNvMB1I_B1G_3nowE0zE0E0Cs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555>:
 3b58590:      	pushq	%rbx
 3b58591:      	movq	(%rdi), %rax
 3b58594:      	movq	(%rax), %rbx
 3b58597:      	movq	$0x0, (%rax)
 3b5859e:      	testq	%rbx, %rbx
 3b585a1:      	je	0x3b585b1 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockNtNtBc_4time7InstantE10initializeNCINvB1a_11get_or_initNvMB1I_B1G_3nowE0zE0E0Cs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555+0x21>
 3b585a3:      	callq	*0x15e8907(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3b585a9:      	movq	%rax, (%rbx)
 3b585ac:      	movl	%edx, 0x8(%rbx)
 3b585af:      	popq	%rbx
 3b585b0:      	retq
 3b585b1:      	leaq	0x14cf998(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b585b8:      	callq	*0x15e8882(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b585be:      	int3
 3b585bf:      	int3

0000000003b5cdf0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555>:
 3b5cdf0:      	pushq	%r14
 3b5cdf2:      	pushq	%rbx
 3b5cdf3:      	subq	$0x28, %rsp
 3b5cdf7:      	movq	(%rdi), %rax
 3b5cdfa:      	movq	(%rax), %rbx
 3b5cdfd:      	movq	$0x0, (%rax)
 3b5ce04:      	testq	%rbx, %rbx
 3b5ce07:      	je	0x3b5cf75 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x185>
 3b5ce0d:      	leaq	-0x2d42e72(%rip), %rsi  # 0xe19fa2 <anon.3222e37afe7119ca9a4f598626aaaddd.303.llvm.15726592429269621555+0x13>
 3b5ce14:      	leaq	0x8(%rsp), %rdi
 3b5ce19:      	movl	$0x1a, %edx
 3b5ce1e:      	callq	*0x15e4024(%rip)        # 0x5140e48 <writev+0x5140e48>
 3b5ce24:      	cmpl	$0x1, 0x8(%rsp)
 3b5ce29:      	jne	0x3b5ce4d <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x5d>
 3b5ce2b:      	movq	0x10(%rsp), %rsi
 3b5ce30:      	cmpq	$-0x1, %rsi
 3b5ce34:      	je	0x3b5ce87 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce36:      	testq	%rsi, %rsi
 3b5ce39:      	je	0x3b5ce87 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce3b:      	movq	0x18(%rsp), %rdi
 3b5ce40:      	movl	$0x1, %edx
 3b5ce45:      	callq	*0x15e3ef5(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b5ce4b:      	jmp	0x3b5ce87 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce4d:      	movq	0x10(%rsp), %rsi
 3b5ce52:      	cmpq	$-0x1, %rsi
 3b5ce56:      	je	0x3b5ce87 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x97>
 3b5ce58:      	movq	0x18(%rsp), %rdi
 3b5ce5d:      	movq	0x20(%rsp), %rcx
 3b5ce62:      	testq	%rcx, %rcx
 3b5ce65:      	je	0x3b5ce8f <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x9f>
 3b5ce67:      	cmpq	$0x1, %rcx
 3b5ce6b:      	jne	0x3b5cea0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0xb0>
 3b5ce6d:      	movzbl	(%rdi), %eax
 3b5ce70:      	xorl	%r14d, %r14d
 3b5ce73:      	cmpl	$0x2b, %eax
 3b5ce76:      	je	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5ce7c:      	cmpl	$0x2d, %eax
 3b5ce7f:      	je	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5ce85:      	jmp	0x3b5cea3 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0xb3>
 3b5ce87:      	xorl	%r14d, %r14d
 3b5ce8a:      	jmp	0x3b5cf60 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5ce8f:      	movq	%rcx, %r14
 3b5ce92:      	testq	%rsi, %rsi
 3b5ce95:      	jne	0x3b5cf55 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x165>
 3b5ce9b:      	jmp	0x3b5cf60 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cea0:      	movzbl	(%rdi), %eax
 3b5cea3:      	xorl	%r8d, %r8d
 3b5cea6:      	cmpb	$0x2b, %al
 3b5cea8:      	sete	%r8b
 3b5ceac:      	movq	%r8, %rax
 3b5ceaf:      	negq	%rax
 3b5ceb2:      	movq	%rcx, %rdx
 3b5ceb5:      	subq	%r8, %rdx
 3b5ceb8:      	addq	%rdi, %r8
 3b5cebb:      	cmpq	$0x11, %rdx
 3b5cebf:      	jae	0x3b5cf04 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x114>
 3b5cec1:      	testq	%rdx, %rdx
 3b5cec4:      	je	0x3b5cf43 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x153>
 3b5cec6:      	addq	%rax, %rcx
 3b5cec9:      	negq	%rcx
 3b5cecc:      	xorl	%eax, %eax
 3b5cece:      	xorl	%r14d, %r14d
 3b5ced1:      	nopw	%cs:(%rax,%rax)
 3b5cee0:      	movzbl	(%r8,%rax), %edx
 3b5cee5:      	addl	$-0x30, %edx
 3b5cee8:      	cmpl	$0x9, %edx
 3b5ceeb:      	ja	0x3b5cf4d <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x15d>
 3b5ceed:      	leaq	(%r14,%r14,4), %r9
 3b5cef1:      	movl	%edx, %edx
 3b5cef3:      	leaq	(%rdx,%r9,2), %r14
 3b5cef7:      	incq	%rax
 3b5cefa:      	movq	%rcx, %rdx
 3b5cefd:      	addq	%rax, %rdx
 3b5cf00:      	jne	0x3b5cee0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0xf0>
 3b5cf02:      	jmp	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf04:      	addq	%rax, %rcx
 3b5cf07:      	negq	%rcx
 3b5cf0a:      	xorl	%r14d, %r14d
 3b5cf0d:      	movl	$0xa, %r9d
 3b5cf13:      	xorl	%r10d, %r10d
 3b5cf16:      	xorl	%eax, %eax
 3b5cf18:      	movq	%rcx, %rdx
 3b5cf1b:      	addq	%r10, %rdx
 3b5cf1e:      	je	0x3b5cf6b <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x17b>
 3b5cf20:      	mulq	%r9
 3b5cf23:      	jo	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf25:      	movzbl	(%r8,%r10), %r11d
 3b5cf2a:      	addl	$-0x30, %r11d
 3b5cf2e:      	addq	%r11, %rax
 3b5cf31:      	setb	%dl
 3b5cf34:      	cmpl	$0x9, %r11d
 3b5cf38:      	ja	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf3a:      	incq	%r10
 3b5cf3d:      	testb	%dl, %dl
 3b5cf3f:      	je	0x3b5cf18 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x128>
 3b5cf41:      	jmp	0x3b5cf50 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x160>
 3b5cf43:      	xorl	%r14d, %r14d
 3b5cf46:      	testq	%rsi, %rsi
 3b5cf49:      	jne	0x3b5cf55 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x165>
 3b5cf4b:      	jmp	0x3b5cf60 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cf4d:      	xorl	%r14d, %r14d
 3b5cf50:      	testq	%rsi, %rsi
 3b5cf53:      	je	0x3b5cf60 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cf55:      	movl	$0x1, %edx
 3b5cf5a:      	callq	*0x15e3de0(%rip)        # 0x5140d40 <writev+0x5140d40>
 3b5cf60:      	movq	%r14, (%rbx)
 3b5cf63:      	addq	$0x28, %rsp
 3b5cf67:      	popq	%rbx
 3b5cf68:      	popq	%r14
 3b5cf6a:      	retq
 3b5cf6b:      	movq	%rax, %r14
 3b5cf6e:      	testq	%rsi, %rsi
 3b5cf71:      	jne	0x3b5cf55 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x165>
 3b5cf73:      	jmp	0x3b5cf60 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555+0x170>
 3b5cf75:      	leaq	0x14cafd4(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b5cf7c:      	callq	*0x15e3ebe(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b5cf82:      	int3
 3b5cf83:      	int3
 3b5cf84:      	int3
 3b5cf85:      	int3
 3b5cf86:      	int3
 3b5cf87:      	int3
 3b5cf88:      	int3
 3b5cf89:      	int3
 3b5cf8a:      	int3
 3b5cf8b:      	int3
 3b5cf8c:      	int3
 3b5cf8d:      	int3
 3b5cf8e:      	int3
 3b5cf8f:      	int3

0000000003b832a0 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockAjj3_E10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB2t_.llvm.15726592429269621555>:
 3b832a0:      	pushq	%r14
 3b832a2:      	pushq	%rbx
 3b832a3:      	subq	$0x48, %rsp
 3b832a7:      	movq	(%rdi), %rax
 3b832aa:      	movq	(%rax), %r14
 3b832ad:      	movq	$0x0, (%rax)
 3b832b4:      	testq	%r14, %r14
 3b832b7:      	je	0x3b832df <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockAjj3_E10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB2t_.llvm.15726592429269621555+0x3f>
 3b832b9:      	callq	0x387b890 <_RNvNtCs3pwlnhBXFtN_12memra_server14route_contract22hybrid_interactive_cap>
 3b832be:      	movq	%rax, %rbx
 3b832c1:      	movq	%rsp, %rdi
 3b832c4:      	callq	*0x15be926(%rip)        # 0x5141bf0 <writev+0x5141bf0>
 3b832ca:      	movups	0x28(%rsp), %xmm0
 3b832cf:      	movq	%rbx, (%r14)
 3b832d2:      	movups	%xmm0, 0x8(%r14)
 3b832d7:      	addq	$0x48, %rsp
 3b832db:      	popq	%rbx
 3b832dc:      	popq	%r14
 3b832de:      	retq
 3b832df:      	leaq	0x14a4c6a(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b832e6:      	callq	*0x15bdb54(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b832ec:      	int3
 3b832ed:      	int3
 3b832ee:      	int3
 3b832ef:      	int3

0000000003b83480 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0INtNtNtB1Q_3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB32_.llvm.15726592429269621555>:
 3b83480:      	pushq	%rax
 3b83481:      	movq	(%rdi), %rax
 3b83484:      	movq	%rax, (%rsp)
 3b83488:      	movq	%rsp, %rdi
 3b8348b:      	callq	0x3b56370 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zE0E0B2X_.llvm.15726592429269621555>
 3b83490:      	popq	%rax
 3b83491:      	retq
 3b83492:      	int3
 3b83493:      	int3
 3b83494:      	int3
 3b83495:      	int3
 3b83496:      	int3
 3b83497:      	int3
 3b83498:      	int3
 3b83499:      	int3
 3b8349a:      	int3
 3b8349b:      	int3
 3b8349c:      	int3
 3b8349d:      	int3
 3b8349e:      	int3
 3b8349f:      	int3

0000000003b838b0 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockNtNtBh_4time7InstantE10initializeNCINvB1f_11get_or_initNvMB1N_B1L_3nowE0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableCs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555>:
 3b838b0:      	pushq	%rbx
 3b838b1:      	movq	(%rdi), %rax
 3b838b4:      	movq	(%rax), %rbx
 3b838b7:      	movq	$0x0, (%rax)
 3b838be:      	testq	%rbx, %rbx
 3b838c1:      	je	0x3b838d1 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockNtNtBh_4time7InstantE10initializeNCINvB1f_11get_or_initNvMB1N_B1L_3nowE0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableCs3pwlnhBXFtN_12memra_server.llvm.15726592429269621555+0x21>
 3b838c3:      	callq	*0x15bd5e7(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3b838c9:      	movq	%rax, (%rbx)
 3b838cc:      	movl	%edx, 0x8(%rbx)
 3b838cf:      	popq	%rbx
 3b838d0:      	retq
 3b838d1:      	leaq	0x14a4678(%rip), %rdi   # 0x5027f50 <anon.3222e37afe7119ca9a4f598626aaaddd.196.llvm.15726592429269621555+0xb0>
 3b838d8:      	callq	*0x15bd562(%rip)        # 0x5140e40 <writev+0x5140e40>
 3b838de:      	int3
 3b838df:      	int3

0000000003b85740 <_RNSNvYNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtBd_4Once15call_once_forceNCINvMNtBf_9once_lockINtB1g_8OnceLockyE10initializeNCINvB1f_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTRNtBd_9OnceStateEE9call_once6vtableB2p_.llvm.15726592429269621555>:
 3b85740:      	pushq	%rax
 3b85741:      	movq	(%rdi), %rax
 3b85744:      	movq	%rax, (%rsp)
 3b85748:      	movq	%rsp, %rdi
 3b8574b:      	callq	0x3b5cdf0 <_RNCINvMs0_NtNtCs2AWtUsOyxgP_3std4sync4onceNtB8_4Once15call_once_forceNCINvMNtBa_9once_lockINtB1b_8OnceLockyE10initializeNCINvB1a_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server20queue_wait_ceiling_s0E0zE0E0B2k_.llvm.15726592429269621555>
 3b85750:      	popq	%rax
 3b85751:      	retq
 3b85752:      	int3
 3b85753:      	int3
 3b85754:      	int3
 3b85755:      	int3
 3b85756:      	int3
 3b85757:      	int3
 3b85758:      	int3
 3b85759:      	int3
 3b8575a:      	int3
 3b8575b:      	int3
 3b8575c:      	int3
 3b8575d:      	int3
 3b8575e:      	int3
 3b8575f:      	int3

0000000003c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEEB1T_.llvm.12140772379724168186>:
 3c03370:      	pushq	%rbx
 3c03371:      	movq	(%rdi), %rbx
 3c03374:      	cmpb	$0x0, 0x8(%rdi)
 3c03378:      	jne	0x3c0338c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x1c>
 3c0337a:      	movq	0x153dce7(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c03381:      	movq	(%rax), %rax
 3c03384:      	shlq	%rax
 3c03387:      	testq	%rax, %rax
 3c0338a:      	jne	0x3c033a1 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x31>
 3c0338c:      	xorl	%eax, %eax
 3c0338e:      	xchgl	%eax, (%rbx)
 3c03390:      	cmpl	$0x2, %eax
 3c03393:      	je	0x3c03397 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x27>
 3c03395:      	popq	%rbx
 3c03396:      	retq
 3c03397:      	movq	%rbx, %rdi
 3c0339a:      	popq	%rbx
 3c0339b:      	jmpq	*0x153dccf(%rip)        # 0x5141070 <writev+0x5141070>
 3c033a1:      	callq	*0x153dcd1(%rip)        # 0x5141078 <writev+0x5141078>
 3c033a7:      	testb	%al, %al
 3c033a9:      	jne	0x3c0338c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x1c>
 3c033ab:      	movb	$0x1, 0x4(%rbx)
 3c033af:      	jmp	0x3c0338c <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186+0x1c>
 3c033b1:      	int3
 3c033b2:      	int3
 3c033b3:      	int3
 3c033b4:      	int3
 3c033b5:      	int3
 3c033b6:      	int3
 3c033b7:      	int3
 3c033b8:      	int3
 3c033b9:      	int3
 3c033ba:      	int3
 3c033bb:      	int3
 3c033bc:      	int3
 3c033bd:      	int3
 3c033be:      	int3
 3c033bf:      	int3

0000000003c736b0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output>:
 3c736b0:      	pushq	%rbp
 3c736b1:      	pushq	%r15
 3c736b3:      	pushq	%r14
 3c736b5:      	pushq	%r13
 3c736b7:      	pushq	%r12
 3c736b9:      	pushq	%rbx
 3c736ba:      	subq	$0x78, %rsp
 3c736be:      	movq	%rcx, %r14
 3c736c1:      	movq	%rdx, %r15
 3c736c4:      	movq	%rsi, %r13
 3c736c7:      	movq	%rdi, %rbx
 3c736ca:      	movl	$0x1, %ecx
 3c736cf:      	xorl	%eax, %eax
 3c736d1:      	lock
 3c736d2:      	cmpxchgl	%ecx, (%rdi)
 3c736d5:      	jne	0x3c73a0f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x35f>
 3c736db:      	movq	0x14cd986(%rip), %r12   # 0x5141068 <writev+0x5141068>
 3c736e2:      	movq	(%r12), %rax
 3c736e6:      	shlq	%rax
 3c736e9:      	testq	%rax, %rax
 3c736ec:      	jne	0x3c73a1d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x36d>
 3c736f2:      	xorl	%esi, %esi
 3c736f4:      	movzbl	0x4(%rbx), %eax
 3c736f8:      	testb	%al, %al
 3c736fa:      	jne	0x3c73a35 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x385>
 3c73700:      	cmpq	$0x0, 0x20(%rbx)
 3c73705:      	je	0x3c7396b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bb>
 3c7370b:      	movl	%esi, 0xc(%rsp)
 3c7370f:      	leaq	0x28(%rbx), %rdi
 3c73713:      	movq	%r13, 0x28(%rsp)
 3c73718:      	movq	%r13, %rsi
 3c7371b:      	movq	%r15, %rdx
 3c7371e:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73723:      	movq	%rax, %r13
 3c73726:      	shrq	$0x39, %rax
 3c7372a:      	movq	0x8(%rbx), %rdx
 3c7372e:      	movq	0x10(%rbx), %rcx
 3c73732:      	movd	%eax, %xmm0
 3c73736:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c7373a:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c7373f:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73744:      	xorl	%esi, %esi
 3c73746:      	pcmpeqd	%xmm2, %xmm2
 3c7374a:      	movq	0x14cd657(%rip), %r8    # 0x5140da8 <writev+0x5140da8>
 3c73751:      	andq	%rcx, %r13
 3c73754:      	movdqu	(%rdx,%r13), %xmm3
 3c7375a:      	movdqa	%xmm3, %xmm0
 3c7375e:      	pcmpeqb	%xmm1, %xmm0
 3c73762:      	pmovmskb	%xmm0, %r12d
 3c73767:      	testl	%r12d, %r12d
 3c7376a:      	je	0x3c73800 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x150>
 3c73770:      	movq	%r14, 0x10(%rsp)
 3c73775:      	movq	%rdx, 0x18(%rsp)
 3c7377a:      	movdqa	%xmm1, 0x50(%rsp)
 3c73780:      	movq	%rsi, 0x40(%rsp)
 3c73785:      	movdqa	%xmm3, 0x30(%rsp)
 3c7378b:      	tzcntl	%r12d, %eax
 3c73790:      	addq	%r13, %rax
 3c73793:      	andq	%rcx, %rax
 3c73796:      	shlq	$0x8, %rax
 3c7379a:      	movq	%rdx, %r14
 3c7379d:      	subq	%rax, %r14
 3c737a0:      	cmpq	-0xf0(%r14), %r15
 3c737a7:      	jne	0x3c737cf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x11f>
 3c737a9:      	movq	-0xf8(%r14), %rsi
 3c737b0:      	movq	0x28(%rsp), %rdi
 3c737b5:      	movq	%r15, %rdx
 3c737b8:      	movq	%rcx, 0x20(%rsp)
 3c737bd:      	movq	%r8, %rbp
 3c737c0:      	callq	*%r8
 3c737c3:      	movq	%rbp, %r8
 3c737c6:      	movq	0x20(%rsp), %rcx
 3c737cb:      	testl	%eax, %eax
 3c737cd:      	je	0x3c73820 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x170>
 3c737cf:      	leal	-0x1(%r12), %eax
 3c737d4:      	andw	%r12w, %ax
 3c737d8:      	movl	%eax, %r12d
 3c737db:      	movq	0x10(%rsp), %r14
 3c737e0:      	movq	0x18(%rsp), %rdx
 3c737e5:      	movdqa	0x50(%rsp), %xmm1
 3c737eb:      	movq	0x40(%rsp), %rsi
 3c737f0:      	pcmpeqd	%xmm2, %xmm2
 3c737f4:      	movdqa	0x30(%rsp), %xmm3
 3c737fa:      	jne	0x3c7378b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0xdb>
 3c737fc:      	nopl	(%rax)
 3c73800:      	pcmpeqb	%xmm2, %xmm3
 3c73804:      	pmovmskb	%xmm3, %eax
 3c73808:      	testl	%eax, %eax
 3c7380a:      	jne	0x3c73974 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2c4>
 3c73810:      	addq	%rsi, %r13
 3c73813:      	addq	$0x10, %r13
 3c73817:      	addq	$0x10, %rsi
 3c7381b:      	jmp	0x3c73751 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0xa1>
 3c73820:      	movb	$0x1, %al
 3c73822:      	cmpb	$0x2, -0x88(%r14)
 3c7382a:      	jae	0x3c73976 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2c6>
 3c73830:      	cmpq	$-0x2, -0x80(%r14)
 3c73835:      	movq	0x14cd82c(%rip), %r12   # 0x5141068 <writev+0x5141068>
 3c7383c:      	movl	0xc(%rsp), %esi
 3c73840:      	jne	0x3c7396d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bd>
 3c73846:      	movq	-0x8(%r14), %rax
 3c7384a:      	movq	0x10(%rsp), %rdi
 3c7384f:      	cmpq	%rax, %rdi
 3c73852:      	cmovbeq	%rax, %rdi
 3c73856:      	movq	0x38(%rbx), %rdx
 3c7385a:      	xorl	%ecx, %ecx
 3c7385c:      	subq	%rax, %rdx
 3c7385f:      	cmovaeq	%rdx, %rcx
 3c73863:      	movb	$0x2, %al
 3c73865:      	movq	%rdi, 0x10(%rsp)
 3c7386a:      	addq	%rdi, %rcx
 3c7386d:      	jb	0x3c7396d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bd>
 3c73873:      	cmpq	0x40(%rbx), %rcx
 3c73877:      	ja	0x3c7396d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2bd>
 3c7387d:      	movq	%r15, %r13
 3c73880:      	movq	%rcx, 0x38(%rbx)
 3c73884:      	leaq	0x28(%rbx), %rdi
 3c73888:      	movq	0x28(%rsp), %rsi
 3c7388d:      	movq	%r15, %rdx
 3c73890:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73895:      	movq	%rax, %rcx
 3c73898:      	shrq	$0x39, %rcx
 3c7389c:      	movd	%ecx, %xmm0
 3c738a0:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c738a4:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c738a9:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c738ae:      	xorl	%r15d, %r15d
 3c738b1:      	pcmpeqd	%xmm2, %xmm2
 3c738b5:      	movq	0x20(%rsp), %rdx
 3c738ba:      	andq	%rdx, %rax
 3c738bd:      	movq	0x18(%rsp), %rcx
 3c738c2:      	movdqu	(%rcx,%rax), %xmm3
 3c738c7:      	movdqa	%xmm3, %xmm0
 3c738cb:      	pcmpeqb	%xmm1, %xmm0
 3c738cf:      	pmovmskb	%xmm0, %r12d
 3c738d4:      	testl	%r12d, %r12d
 3c738d7:      	je	0x3c7394b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x29b>
 3c738d9:      	movq	%rax, 0x50(%rsp)
 3c738de:      	movdqa	%xmm1, 0x40(%rsp)
 3c738e4:      	movdqa	%xmm3, 0x30(%rsp)
 3c738ea:      	tzcntl	%r12d, %ecx
 3c738ef:      	addq	%rax, %rcx
 3c738f2:      	andq	%rdx, %rcx
 3c738f5:      	shlq	$0x8, %rcx
 3c738f9:      	movq	0x18(%rsp), %r14
 3c738fe:      	subq	%rcx, %r14
 3c73901:      	cmpq	-0xf0(%r14), %r13
 3c73908:      	jne	0x3c73923 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x273>
 3c7390a:      	movq	%r13, %rdx
 3c7390d:      	movq	-0xf8(%r14), %rsi
 3c73914:      	movq	0x28(%rsp), %rdi
 3c73919:      	callq	*%rbp
 3c7391b:      	testl	%eax, %eax
 3c7391d:      	je	0x3c739b5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x305>
 3c73923:      	leal	-0x1(%r12), %eax
 3c73928:      	andw	%r12w, %ax
 3c7392c:      	movl	%eax, %r12d
 3c7392f:      	movq	0x50(%rsp), %rax
 3c73934:      	movq	0x20(%rsp), %rdx
 3c73939:      	movdqa	0x40(%rsp), %xmm1
 3c7393f:      	pcmpeqd	%xmm2, %xmm2
 3c73943:      	movdqa	0x30(%rsp), %xmm3
 3c73949:      	jne	0x3c738ea <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x23a>
 3c7394b:      	pcmpeqb	%xmm2, %xmm3
 3c7394f:      	pmovmskb	%xmm3, %ecx
 3c73953:      	testl	%ecx, %ecx
 3c73955:      	jne	0x3c739fc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x34c>
 3c7395b:      	addq	%r15, %rax
 3c7395e:      	addq	$0x10, %rax
 3c73962:      	addq	$0x10, %r15
 3c73966:      	jmp	0x3c738b5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x205>
 3c7396b:      	xorl	%eax, %eax
 3c7396d:      	testb	%sil, %sil
 3c73970:      	je	0x3c73986 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2d6>
 3c73972:      	jmp	0x3c7399d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73974:      	xorl	%eax, %eax
 3c73976:      	movq	0x14cd6eb(%rip), %r12   # 0x5141068 <writev+0x5141068>
 3c7397d:      	movl	0xc(%rsp), %esi
 3c73981:      	testb	%sil, %sil
 3c73984:      	jne	0x3c7399d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73986:      	movq	(%r12), %rcx
 3c7398a:      	movabsq	$0x7fffffffffffffff, %rdx # imm = 0x7FFFFFFFFFFFFFFF
 3c73994:      	testq	%rdx, %rcx
 3c73997:      	jne	0x3c73a66 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x3b6>
 3c7399d:      	xorl	%ecx, %ecx
 3c7399f:      	xchgl	%ecx, (%rbx)
 3c739a1:      	cmpl	$0x2, %ecx
 3c739a4:      	je	0x3c739ed <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x33d>
 3c739a6:      	addq	$0x78, %rsp
 3c739aa:      	popq	%rbx
 3c739ab:      	popq	%r12
 3c739ad:      	popq	%r13
 3c739af:      	popq	%r14
 3c739b1:      	popq	%r15
 3c739b3:      	popq	%rbp
 3c739b4:      	retq
 3c739b5:      	movq	0x10(%rsp), %rax
 3c739ba:      	movq	%rax, -0x8(%r14)
 3c739be:      	cmpb	$0x0, 0xc(%rsp)
 3c739c3:      	jne	0x3c739e2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x332>
 3c739c5:      	movq	0x14cd69c(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c739cc:      	movq	(%rax), %rax
 3c739cf:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c739d9:      	testq	%rcx, %rax
 3c739dc:      	jne	0x3c73a83 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x3d3>
 3c739e2:      	xorl	%ecx, %ecx
 3c739e4:      	xchgl	%ecx, (%rbx)
 3c739e6:      	movb	$-0x1, %al
 3c739e8:      	cmpl	$0x2, %ecx
 3c739eb:      	jne	0x3c739a6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2f6>
 3c739ed:      	movq	%rbx, %rdi
 3c739f0:      	movl	%eax, %ebx
 3c739f2:      	callq	*0x14cd678(%rip)        # 0x5141070 <writev+0x5141070>
 3c739f8:      	movl	%ebx, %eax
 3c739fa:      	jmp	0x3c739a6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2f6>
 3c739fc:      	leaq	0x13b900d(%rip), %rdi   # 0x502ca10 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x250>
 3c73a03:      	movl	0xc(%rsp), %ebp
 3c73a07:      	callq	*0x14cd433(%rip)        # 0x5140e40 <writev+0x5140e40>
 3c73a0d:      	jmp	0x3c73a64 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x3b4>
 3c73a0f:      	movq	%rbx, %rdi
 3c73a12:      	callq	*0x14cd9f0(%rip)        # 0x5141408 <writev+0x5141408>
 3c73a18:      	jmp	0x3c736db <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2b>
 3c73a1d:      	callq	*0x14cd655(%rip)        # 0x5141078 <writev+0x5141078>
 3c73a23:      	movl	%eax, %esi
 3c73a25:      	xorb	$0x1, %sil
 3c73a29:      	movzbl	0x4(%rbx), %eax
 3c73a2d:      	testb	%al, %al
 3c73a2f:      	je	0x3c73700 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x50>
 3c73a35:      	movq	%rbx, 0x68(%rsp)
 3c73a3a:      	movb	%sil, 0x70(%rsp)
 3c73a3f:      	leaq	-0x2e5501d(%rip), %rdi  # 0xe1ea29 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c73a46:      	leaq	0x13b8aa3(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c73a4d:      	leaq	0x13b8fa4(%rip), %r8    # 0x502c9f8 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x238>
 3c73a54:      	leaq	0x68(%rsp), %rdx
 3c73a59:      	movl	$0x2b, %esi
 3c73a5e:      	callq	*0x14cd434(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c73a64:      	ud2
 3c73a66:      	movl	%eax, %ebp
 3c73a68:      	callq	*0x14cd60a(%rip)        # 0x5141078 <writev+0x5141078>
 3c73a6e:      	movl	%eax, %ecx
 3c73a70:      	movl	%ebp, %eax
 3c73a72:      	testb	%cl, %cl
 3c73a74:      	jne	0x3c7399d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73a7a:      	movb	$0x1, 0x4(%rbx)
 3c73a7e:      	jmp	0x3c7399d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x2ed>
 3c73a83:      	callq	*0x14cd5ef(%rip)        # 0x5141078 <writev+0x5141078>
 3c73a89:      	testb	%al, %al
 3c73a8b:      	jne	0x3c739e2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x332>
 3c73a91:      	movb	$0x1, 0x4(%rbx)
 3c73a95:      	jmp	0x3c739e2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x332>
 3c73a9a:      	movq	%rax, %r14
 3c73a9d:      	leaq	0x68(%rsp), %rdi
 3c73aa2:      	callq	0x3c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c73aa7:      	jmp	0x3c73abe <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore14reserve_output+0x40e>
 3c73aa9:      	callq	*0x14cd309(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c73aaf:      	movq	%rax, %r14
 3c73ab2:      	movzbl	%bpl, %esi
 3c73ab6:      	movq	%rbx, %rdi
 3c73ab9:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c73abe:      	movq	%r14, %rdi
 3c73ac1:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c73ac6:      	callq	*0x14cd2ec(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c73acc:      	int3
 3c73acd:      	int3
 3c73ace:      	int3
 3c73acf:      	int3

0000000003c73ad0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal>:
 3c73ad0:      	pushq	%rbp
 3c73ad1:      	pushq	%r15
 3c73ad3:      	pushq	%r14
 3c73ad5:      	pushq	%r13
 3c73ad7:      	pushq	%r12
 3c73ad9:      	pushq	%rbx
 3c73ada:      	subq	$0x128, %rsp            # imm = 0x128
 3c73ae1:      	movq	%r9, 0x18(%rsp)
 3c73ae6:      	movq	%r8, 0x38(%rsp)
 3c73aeb:      	movq	%rcx, %r13
 3c73aee:      	movb	$0x3, %r14b
 3c73af1:      	cmpb	$0x2, 0x60(%rcx)
 3c73af5:      	jb	0x3c73dfc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x32c>
 3c73afb:      	movq	%rdx, %r15
 3c73afe:      	movq	%rsi, %r12
 3c73b01:      	movq	%rdi, %rbx
 3c73b04:      	movq	%r13, %rdi
 3c73b07:      	callq	0x3c6efc0 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store17record_size_bytes>
 3c73b0c:      	movq	%rax, %rbp
 3c73b0f:      	movl	$0x1, %ecx
 3c73b14:      	xorl	%eax, %eax
 3c73b16:      	lock
 3c73b17:      	cmpxchgl	%ecx, (%rbx)
 3c73b1a:      	jne	0x3c7421f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x74f>
 3c73b20:      	movq	0x14cd541(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c73b27:      	movq	(%rax), %rax
 3c73b2a:      	shlq	%rax
 3c73b2d:      	testq	%rax, %rax
 3c73b30:      	jne	0x3c7422d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x75d>
 3c73b36:      	movl	$0x0, 0x20(%rsp)
 3c73b3e:      	movzbl	0x4(%rbx), %eax
 3c73b42:      	testb	%al, %al
 3c73b44:      	jne	0x3c74245 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x775>
 3c73b4a:      	cmpq	$0x0, 0x20(%rbx)
 3c73b4f:      	je	0x3c73dc8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2f8>
 3c73b55:      	leaq	0x28(%rbx), %rdi
 3c73b59:      	movq	%r12, 0x10(%rsp)
 3c73b5e:      	movq	%r12, %rsi
 3c73b61:      	movq	%r15, 0x8(%rsp)
 3c73b66:      	movq	%r15, %rdx
 3c73b69:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73b6e:      	movq	%rax, %rcx
 3c73b71:      	shrq	$0x39, %rcx
 3c73b75:      	movq	0x8(%rbx), %r15
 3c73b79:      	movq	0x10(%rbx), %r12
 3c73b7d:      	movd	%ecx, %xmm0
 3c73b81:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73b85:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73b8a:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73b8f:      	xorl	%edx, %edx
 3c73b91:      	pcmpeqd	%xmm2, %xmm2
 3c73b95:      	andq	%r12, %rax
 3c73b98:      	movdqu	(%r15,%rax), %xmm3
 3c73b9e:      	movdqa	%xmm3, %xmm0
 3c73ba2:      	pcmpeqb	%xmm1, %xmm0
 3c73ba6:      	pmovmskb	%xmm0, %r14d
 3c73bab:      	testl	%r14d, %r14d
 3c73bae:      	je	0x3c73c46 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x176>
 3c73bb4:      	movq	%rax, 0x40(%rsp)
 3c73bb9:      	movq	%rbp, 0x30(%rsp)
 3c73bbe:      	movdqa	%xmm1, 0xf0(%rsp)
 3c73bc7:      	movq	%rdx, 0xe0(%rsp)
 3c73bcf:      	movdqa	%xmm3, 0xd0(%rsp)
 3c73bd8:      	tzcntl	%r14d, %ecx
 3c73bdd:      	addq	%rax, %rcx
 3c73be0:      	andq	%r12, %rcx
 3c73be3:      	shlq	$0x8, %rcx
 3c73be7:      	movq	%r15, %rbp
 3c73bea:      	subq	%rcx, %rbp
 3c73bed:      	movq	0x8(%rsp), %rdx
 3c73bf2:      	cmpq	-0xf0(%rbp), %rdx
 3c73bf9:      	jne	0x3c73c11 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x141>
 3c73bfb:      	movq	-0xf8(%rbp), %rsi
 3c73c02:      	movq	0x10(%rsp), %rdi
 3c73c07:      	callq	*0x14cd19b(%rip)        # 0x5140da8 <writev+0x5140da8>
 3c73c0d:      	testl	%eax, %eax
 3c73c0f:      	je	0x3c73c66 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x196>
 3c73c11:      	leal	-0x1(%r14), %eax
 3c73c15:      	andw	%r14w, %ax
 3c73c19:      	movl	%eax, %r14d
 3c73c1c:      	movq	0x40(%rsp), %rax
 3c73c21:      	movq	0x30(%rsp), %rbp
 3c73c26:      	movdqa	0xf0(%rsp), %xmm1
 3c73c2f:      	movq	0xe0(%rsp), %rdx
 3c73c37:      	pcmpeqd	%xmm2, %xmm2
 3c73c3b:      	movdqa	0xd0(%rsp), %xmm3
 3c73c44:      	jne	0x3c73bd8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x108>
 3c73c46:      	pcmpeqb	%xmm2, %xmm3
 3c73c4a:      	pmovmskb	%xmm3, %ecx
 3c73c4e:      	testl	%ecx, %ecx
 3c73c50:      	jne	0x3c73e3f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x36f>
 3c73c56:      	addq	%rdx, %rax
 3c73c59:      	addq	$0x10, %rax
 3c73c5d:      	addq	$0x10, %rdx
 3c73c61:      	jmp	0x3c73b95 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xc5>
 3c73c66:      	movb	$0x1, %r14b
 3c73c69:      	cmpb	$0x2, -0x88(%rbp)
 3c73c70:      	jae	0x3c73dcb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73c76:      	cmpq	$-0x2, -0x80(%rbp)
 3c73c7b:      	jne	0x3c73dcb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73c81:      	movq	-0x8(%rbp), %rcx
 3c73c85:      	movq	0x30(%rsp), %rax
 3c73c8a:      	cmpq	%rcx, %rax
 3c73c8d:      	movq	%rcx, %rbp
 3c73c90:      	cmovaq	%rax, %rbp
 3c73c94:      	movq	0x38(%rbx), %rdx
 3c73c98:      	xorl	%eax, %eax
 3c73c9a:      	subq	%rcx, %rdx
 3c73c9d:      	cmovaeq	%rdx, %rax
 3c73ca1:      	movb	$0x2, %r14b
 3c73ca4:      	addq	%rbp, %rax
 3c73ca7:      	jb	0x3c73dcb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73cad:      	cmpq	0x40(%rbx), %rax
 3c73cb1:      	ja	0x3c73dcb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2fb>
 3c73cb7:      	movq	%rax, 0x38(%rbx)
 3c73cbb:      	leaq	0x28(%rbx), %rdi
 3c73cbf:      	movq	0x10(%rsp), %rsi
 3c73cc4:      	movq	0x8(%rsp), %rdx
 3c73cc9:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73cce:      	movq	%rax, %rcx
 3c73cd1:      	shrq	$0x39, %rcx
 3c73cd5:      	movd	%ecx, %xmm0
 3c73cd9:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73cdd:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73ce2:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73ce7:      	xorl	%edx, %edx
 3c73ce9:      	pcmpeqd	%xmm2, %xmm2
 3c73ced:      	andq	%r12, %rax
 3c73cf0:      	movdqu	(%r15,%rax), %xmm3
 3c73cf6:      	movdqa	%xmm3, %xmm0
 3c73cfa:      	pcmpeqb	%xmm1, %xmm0
 3c73cfe:      	pmovmskb	%xmm0, %r14d
 3c73d03:      	testl	%r14d, %r14d
 3c73d06:      	je	0x3c73da8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2d8>
 3c73d0c:      	movq	%rax, 0xf0(%rsp)
 3c73d14:      	movq	%rbp, 0x40(%rsp)
 3c73d19:      	movdqa	%xmm1, 0xe0(%rsp)
 3c73d22:      	movq	%rdx, 0xd0(%rsp)
 3c73d2a:      	movdqa	%xmm3, 0x100(%rsp)
 3c73d33:      	tzcntl	%r14d, %ecx
 3c73d38:      	addq	%rax, %rcx
 3c73d3b:      	andq	%r12, %rcx
 3c73d3e:      	shlq	$0x8, %rcx
 3c73d42:      	movq	%r15, %rbp
 3c73d45:      	subq	%rcx, %rbp
 3c73d48:      	movq	0x8(%rsp), %rdx
 3c73d4d:      	cmpq	-0xf0(%rbp), %rdx
 3c73d54:      	jne	0x3c73d70 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x2a0>
 3c73d56:      	movq	-0xf8(%rbp), %rsi
 3c73d5d:      	movq	0x10(%rsp), %rdi
 3c73d62:      	callq	*0x14cd040(%rip)        # 0x5140da8 <writev+0x5140da8>
 3c73d68:      	testl	%eax, %eax
 3c73d6a:      	je	0x3c73e4b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x37b>
 3c73d70:      	leal	-0x1(%r14), %eax
 3c73d74:      	andw	%r14w, %ax
 3c73d78:      	movl	%eax, %r14d
 3c73d7b:      	movq	0xf0(%rsp), %rax
 3c73d83:      	movq	0x40(%rsp), %rbp
 3c73d88:      	movdqa	0xe0(%rsp), %xmm1
 3c73d91:      	movq	0xd0(%rsp), %rdx
 3c73d99:      	pcmpeqd	%xmm2, %xmm2
 3c73d9d:      	movdqa	0x100(%rsp), %xmm3
 3c73da6:      	jne	0x3c73d33 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x263>
 3c73da8:      	pcmpeqb	%xmm2, %xmm3
 3c73dac:      	pmovmskb	%xmm3, %ecx
 3c73db0:      	testl	%ecx, %ecx
 3c73db2:      	jne	0x3c7413b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x66b>
 3c73db8:      	addq	%rdx, %rax
 3c73dbb:      	addq	$0x10, %rax
 3c73dbf:      	addq	$0x10, %rdx
 3c73dc3:      	jmp	0x3c73ced <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x21d>
 3c73dc8:      	xorl	%r14d, %r14d
 3c73dcb:      	cmpb	$0x0, 0x20(%rsp)
 3c73dd0:      	jne	0x3c73def <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c73dd2:      	movq	0x14cd28f(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c73dd9:      	movq	(%rax), %rax
 3c73ddc:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c73de6:      	testq	%rcx, %rax
 3c73de9:      	jne	0x3c74287 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7b7>
 3c73def:      	xorl	%eax, %eax
 3c73df1:      	xchgl	%eax, (%rbx)
 3c73df3:      	cmpl	$0x2, %eax
 3c73df6:      	je	0x3c74279 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7a9>
 3c73dfc:      	cmpq	$-0x1, 0x18(%r13)
 3c73e01:      	je	0x3c73e0c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x33c>
 3c73e03:      	leaq	0x18(%r13), %rdi
 3c73e07:      	callq	0x3c08af0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c73e0c:      	movq	(%r13), %rsi
 3c73e10:      	cmpq	$-0x1, %rsi
 3c73e14:      	je	0x3c73e2a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c73e16:      	testq	%rsi, %rsi
 3c73e19:      	je	0x3c73e2a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c73e1b:      	movq	0x8(%r13), %rdi
 3c73e1f:      	movl	$0x1, %edx
 3c73e24:      	callq	*0x14ccf16(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c73e2a:      	movl	%r14d, %eax
 3c73e2d:      	addq	$0x128, %rsp            # imm = 0x128
 3c73e34:      	popq	%rbx
 3c73e35:      	popq	%r12
 3c73e37:      	popq	%r13
 3c73e39:      	popq	%r14
 3c73e3b:      	popq	%r15
 3c73e3d:      	popq	%rbp
 3c73e3e:      	retq
 3c73e3f:      	xorl	%r14d, %r14d
 3c73e42:      	cmpb	$0x0, 0x20(%rsp)
 3c73e47:      	je	0x3c73dd2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x302>
 3c73e49:      	jmp	0x3c73def <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c73e4b:      	movq	0x40(%rsp), %rax
 3c73e50:      	movq	%rax, -0x8(%rbp)
 3c73e54:      	leaq	-0x80(%rbp), %r14
 3c73e58:      	cmpq	$-0x2, -0x80(%rbp)
 3c73e5d:      	je	0x3c73e67 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x397>
 3c73e5f:      	movq	%r14, %rdi
 3c73e62:      	callq	0x3c08980 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c73e67:      	movq	0x60(%r13), %rax
 3c73e6b:      	movq	%rax, 0x60(%r14)
 3c73e6f:      	movups	0x50(%r13), %xmm0
 3c73e74:      	movups	%xmm0, 0x50(%r14)
 3c73e79:      	movups	0x40(%r13), %xmm0
 3c73e7e:      	movups	%xmm0, 0x40(%r14)
 3c73e83:      	movdqu	(%r13), %xmm0
 3c73e89:      	movdqu	0x10(%r13), %xmm1
 3c73e8f:      	movdqu	0x20(%r13), %xmm2
 3c73e95:      	movdqu	0x30(%r13), %xmm3
 3c73e9b:      	movdqu	%xmm3, 0x30(%r14)
 3c73ea1:      	movdqu	%xmm2, 0x20(%r14)
 3c73ea7:      	movdqu	%xmm1, 0x10(%r14)
 3c73ead:      	movdqu	%xmm0, (%r14)
 3c73eb2:      	cmpb	$0x0, 0x20(%rsp)
 3c73eb7:      	movq	0x14cd1aa(%rip), %r15   # 0x5141068 <writev+0x5141068>
 3c73ebe:      	movq	0x8(%rsp), %r12
 3c73ec3:      	jne	0x3c73edb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x40b>
 3c73ec5:      	movq	(%r15), %rax
 3c73ec8:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c73ed2:      	testq	%rcx, %rax
 3c73ed5:      	jne	0x3c7430b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x83b>
 3c73edb:      	xorl	%eax, %eax
 3c73edd:      	xchgl	%eax, (%rbx)
 3c73edf:      	cmpl	$0x2, %eax
 3c73ee2:      	je	0x3c7429e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7ce>
 3c73ee8:      	leaq	0x110(%rsp), %rdi
 3c73ef0:      	movq	0x38(%rsp), %rsi
 3c73ef5:      	movq	0x18(%rsp), %rax
 3c73efa:      	callq	*0x20(%rax)
 3c73efd:      	movq	0x120(%rsp), %rax
 3c73f05:      	movdqa	0x110(%rsp), %xmm0
 3c73f0e:      	movdqa	%xmm0, 0x50(%rsp)
 3c73f14:      	movq	%rax, 0x60(%rsp)
 3c73f19:      	movl	$0x1, %ecx
 3c73f1e:      	xorl	%eax, %eax
 3c73f20:      	lock
 3c73f21:      	cmpxchgl	%ecx, (%rbx)
 3c73f24:      	jne	0x3c74438 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x968>
 3c73f2a:      	movq	(%r15), %rax
 3c73f2d:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c73f37:      	testq	%rcx, %rax
 3c73f3a:      	jne	0x3c742ac <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7dc>
 3c73f40:      	xorl	%ecx, %ecx
 3c73f42:      	movzbl	0x4(%rbx), %eax
 3c73f46:      	testb	%al, %al
 3c73f48:      	jne	0x3c742c3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7f3>
 3c73f4e:      	leaq	0x13b8b1b(%rip), %r15   # 0x502ca70 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2b0>
 3c73f55:      	movl	$0x22, %esi
 3c73f5a:      	leaq	-0x2e5526c(%rip), %rdi  # 0xe1ecf5 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.296.llvm.12140772379724168186+0x1b5>
 3c73f61:      	cmpq	$0x0, 0x20(%rbx)
 3c73f66:      	movl	%ecx, 0x18(%rsp)
 3c73f6a:      	je	0x3c74159 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x689>
 3c73f70:      	leaq	0x28(%rbx), %rdi
 3c73f74:      	movq	0x10(%rsp), %rsi
 3c73f79:      	movq	%r12, %rdx
 3c73f7c:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c73f81:      	movq	%rax, %r14
 3c73f84:      	shrq	$0x39, %rax
 3c73f88:      	movq	0x8(%rbx), %rbp
 3c73f8c:      	movq	0x10(%rbx), %rcx
 3c73f90:      	movd	%eax, %xmm0
 3c73f94:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c73f98:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c73f9d:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c73fa2:      	xorl	%edx, %edx
 3c73fa4:      	pcmpeqd	%xmm2, %xmm2
 3c73fa8:      	andq	%rcx, %r14
 3c73fab:      	movdqu	(%rbp,%r14), %xmm3
 3c73fb2:      	movdqa	%xmm3, %xmm0
 3c73fb6:      	pcmpeqb	%xmm1, %xmm0
 3c73fba:      	pmovmskb	%xmm0, %r13d
 3c73fbf:      	testl	%r13d, %r13d
 3c73fc2:      	je	0x3c7404b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x57b>
 3c73fc8:      	movq	%rcx, 0x38(%rsp)
 3c73fcd:      	movdqa	%xmm1, 0x20(%rsp)
 3c73fd3:      	movq	%rdx, 0x8(%rsp)
 3c73fd8:      	movdqa	%xmm3, 0x40(%rsp)
 3c73fde:      	tzcntl	%r13d, %eax
 3c73fe3:      	addq	%r14, %rax
 3c73fe6:      	andq	%rcx, %rax
 3c73fe9:      	shlq	$0x8, %rax
 3c73fed:      	movq	%r12, %r15
 3c73ff0:      	movq	%rbp, %r12
 3c73ff3:      	subq	%rax, %r12
 3c73ff6:      	cmpq	-0xf0(%r12), %r15
 3c73ffe:      	jne	0x3c7401a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x54a>
 3c74000:      	movq	-0xf8(%r12), %rsi
 3c74008:      	movq	0x10(%rsp), %rdi
 3c7400d:      	movq	%r15, %rdx
 3c74010:      	callq	*0x14ccd92(%rip)        # 0x5140da8 <writev+0x5140da8>
 3c74016:      	testl	%eax, %eax
 3c74018:      	je	0x3c7406b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x59b>
 3c7401a:      	leal	-0x1(%r13), %eax
 3c7401e:      	andw	%r13w, %ax
 3c74022:      	movl	%eax, %r13d
 3c74025:      	movq	%r15, %r12
 3c74028:      	leaq	0x13b8a41(%rip), %r15   # 0x502ca70 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2b0>
 3c7402f:      	movq	0x38(%rsp), %rcx
 3c74034:      	movdqa	0x20(%rsp), %xmm1
 3c7403a:      	movq	0x8(%rsp), %rdx
 3c7403f:      	pcmpeqd	%xmm2, %xmm2
 3c74043:      	movdqa	0x40(%rsp), %xmm3
 3c74049:      	jne	0x3c73fde <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x50e>
 3c7404b:      	pcmpeqb	%xmm2, %xmm3
 3c7404f:      	pmovmskb	%xmm3, %eax
 3c74053:      	testl	%eax, %eax
 3c74055:      	jne	0x3c7414d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x67d>
 3c7405b:      	addq	%rdx, %r14
 3c7405e:      	addq	$0x10, %r14
 3c74062:      	addq	$0x10, %rdx
 3c74066:      	jmp	0x3c73fa8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x4d8>
 3c7406b:      	movq	-0x80(%r12), %rax
 3c74070:      	movq	$-0x2, -0x80(%r12)
 3c74079:      	cmpq	$-0x2, %rax
 3c7407d:      	je	0x3c742f3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x823>
 3c74083:      	movq	%rax, 0x68(%rsp)
 3c74088:      	movups	-0x78(%r12), %xmm0
 3c7408e:      	movdqu	-0x68(%r12), %xmm1
 3c74095:      	movdqu	-0x58(%r12), %xmm2
 3c7409c:      	movdqu	-0x48(%r12), %xmm3
 3c740a3:      	movups	%xmm0, 0x70(%rsp)
 3c740a8:      	movdqu	%xmm1, 0x80(%rsp)
 3c740b1:      	movdqu	%xmm2, 0x90(%rsp)
 3c740ba:      	movdqu	%xmm3, 0xa0(%rsp)
 3c740c3:      	movups	-0x38(%r12), %xmm0
 3c740c9:      	movups	%xmm0, 0xb0(%rsp)
 3c740d1:      	movdqu	-0x28(%r12), %xmm0
 3c740d8:      	movdqu	%xmm0, 0xc0(%rsp)
 3c740e1:      	cmpq	$-0x1, 0x50(%rsp)
 3c740e7:      	movq	0x30(%rsp), %r13
 3c740ec:      	movq	0x14ccf75(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c740f3:      	je	0x3c74167 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x697>
 3c740f5:      	leaq	0x68(%rsp), %rdi
 3c740fa:      	callq	0x3c08980 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c740ff:      	cmpb	$0x0, 0x18(%rsp)
 3c74104:      	jne	0x3c7411c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x64c>
 3c74106:      	movq	(%r14), %rax
 3c74109:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c74113:      	testq	%rcx, %rax
 3c74116:      	jne	0x3c74330 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x860>
 3c7411c:      	xorl	%eax, %eax
 3c7411e:      	xchgl	%eax, (%rbx)
 3c74120:      	cmpl	$0x2, %eax
 3c74123:      	je	0x3c74322 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x852>
 3c74129:      	leaq	0x50(%rsp), %rdi
 3c7412e:      	callq	0x3bfff70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultIBC_uNtNtCscdodAO9FK5_5alloc6string6StringEINtNtB16_5boxed3BoxDNtNtB4_3any3AnyNtNtB4_6marker4SendEL_EEECs3pwlnhBXFtN_12memra_server>
 3c74133:      	movb	$0x4, %r14b
 3c74136:      	jmp	0x3c73e2a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c7413b:      	leaq	0x13b88fe(%rip), %rdi   # 0x502ca40 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x280>
 3c74142:      	callq	*0x14cccf8(%rip)        # 0x5140e40 <writev+0x5140e40>
 3c74148:      	jmp	0x3c742f1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x821>
 3c7414d:      	movl	$0x22, %esi
 3c74152:      	leaq	-0x2e55464(%rip), %rdi  # 0xe1ecf5 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.296.llvm.12140772379724168186+0x1b5>
 3c74159:      	movq	%r15, %rdx
 3c7415c:      	callq	*0x14ccc96(%rip)        # 0x5140df8 <writev+0x5140df8>
 3c74162:      	jmp	0x3c742f1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x821>
 3c74167:      	leaq	-0xe8(%r12), %r14
 3c7416f:      	movq	-0x8(%r12), %r15
 3c74174:      	movq	%r14, %rdi
 3c74177:      	callq	0x3c08980 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c7417c:      	movq	0xc8(%rsp), %rax
 3c74184:      	movq	%rax, 0x60(%r14)
 3c74188:      	movups	0xb8(%rsp), %xmm0
 3c74190:      	movups	%xmm0, 0x50(%r14)
 3c74195:      	movups	0xa8(%rsp), %xmm0
 3c7419d:      	movups	%xmm0, 0x40(%r14)
 3c741a2:      	movdqu	0x68(%rsp), %xmm0
 3c741a8:      	movdqu	0x78(%rsp), %xmm1
 3c741ae:      	movdqu	0x88(%rsp), %xmm2
 3c741b7:      	movdqu	0x98(%rsp), %xmm3
 3c741c0:      	movdqu	%xmm3, 0x30(%r14)
 3c741c6:      	movdqu	%xmm2, 0x20(%r14)
 3c741cc:      	movdqu	%xmm1, 0x10(%r14)
 3c741d2:      	movdqu	%xmm0, (%r14)
 3c741d7:      	callq	*0x14cccd3(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3c741dd:      	movq	%rax, -0x18(%r12)
 3c741e2:      	movl	%edx, -0x10(%r12)
 3c741e7:      	movq	%r13, -0x8(%r12)
 3c741ec:      	movq	0x38(%rbx), %rax
 3c741f0:      	xorl	%ecx, %ecx
 3c741f2:      	subq	%r15, %rax
 3c741f5:      	cmovaeq	%rax, %rcx
 3c741f9:      	addq	%r13, %rcx
 3c741fc:      	movq	%rcx, 0x38(%rbx)
 3c74200:      	movzbl	0x18(%rsp), %esi
 3c74205:      	movq	%rbx, %rdi
 3c74208:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c7420d:      	leaq	0x50(%rsp), %rdi
 3c74212:      	callq	0x3bfff70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultIBC_uNtNtCscdodAO9FK5_5alloc6string6StringEINtNtB16_5boxed3BoxDNtNtB4_3any3AnyNtNtB4_6marker4SendEL_EEECs3pwlnhBXFtN_12memra_server>
 3c74217:      	movb	$-0x1, %r14b
 3c7421a:      	jmp	0x3c73e2a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x35a>
 3c7421f:      	movq	%rbx, %rdi
 3c74222:      	callq	*0x14cd1e0(%rip)        # 0x5141408 <writev+0x5141408>
 3c74228:      	jmp	0x3c73b20 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x50>
 3c7422d:      	callq	*0x14cce45(%rip)        # 0x5141078 <writev+0x5141078>
 3c74233:      	xorb	$0x1, %al
 3c74235:      	movl	%eax, 0x20(%rsp)
 3c74239:      	movzbl	0x4(%rbx), %eax
 3c7423d:      	testb	%al, %al
 3c7423f:      	je	0x3c73b4a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x7a>
 3c74245:      	movq	%rbx, 0x68(%rsp)
 3c7424a:      	movl	0x20(%rsp), %eax
 3c7424e:      	movb	%al, 0x70(%rsp)
 3c74252:      	leaq	-0x2e55830(%rip), %rdi  # 0xe1ea29 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c74259:      	leaq	0x13b8290(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c74260:      	leaq	0x13b87c1(%rip), %r8    # 0x502ca28 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x268>
 3c74267:      	leaq	0x68(%rsp), %rdx
 3c7426c:      	movl	$0x2b, %esi
 3c74271:      	callq	*0x14ccc21(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c74277:      	jmp	0x3c742f1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x821>
 3c74279:      	movq	%rbx, %rdi
 3c7427c:      	callq	*0x14ccdee(%rip)        # 0x5141070 <writev+0x5141070>
 3c74282:      	jmp	0x3c73dfc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x32c>
 3c74287:      	callq	*0x14ccdeb(%rip)        # 0x5141078 <writev+0x5141078>
 3c7428d:      	testb	%al, %al
 3c7428f:      	jne	0x3c73def <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c74295:      	movb	$0x1, 0x4(%rbx)
 3c74299:      	jmp	0x3c73def <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x31f>
 3c7429e:      	movq	%rbx, %rdi
 3c742a1:      	callq	*0x14ccdc9(%rip)        # 0x5141070 <writev+0x5141070>
 3c742a7:      	jmp	0x3c73ee8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x418>
 3c742ac:      	callq	*0x14ccdc6(%rip)        # 0x5141078 <writev+0x5141078>
 3c742b2:      	movl	%eax, %ecx
 3c742b4:      	xorb	$0x1, %cl
 3c742b7:      	movzbl	0x4(%rbx), %eax
 3c742bb:      	testb	%al, %al
 3c742bd:      	je	0x3c73f4e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x47e>
 3c742c3:      	movq	%rbx, 0x68(%rsp)
 3c742c8:      	movb	%cl, 0x70(%rsp)
 3c742cc:      	leaq	-0x2e558aa(%rip), %rdi  # 0xe1ea29 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c742d3:      	leaq	0x13b8216(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c742da:      	leaq	0x13b8777(%rip), %r8    # 0x502ca58 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x298>
 3c742e1:      	leaq	0x68(%rsp), %rdx
 3c742e6:      	movl	$0x2b, %esi
 3c742eb:      	callq	*0x14ccba7(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c742f1:      	ud2
 3c742f3:      	leaq	0x13b878e(%rip), %r15   # 0x502ca88 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2c8>
 3c742fa:      	movl	$0x1a, %esi
 3c742ff:      	leaq	-0x2e555ef(%rip), %rdi  # 0xe1ed17 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.296.llvm.12140772379724168186+0x1d7>
 3c74306:      	jmp	0x3c74159 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x689>
 3c7430b:      	callq	*0x14ccd67(%rip)        # 0x5141078 <writev+0x5141078>
 3c74311:      	testb	%al, %al
 3c74313:      	jne	0x3c73edb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x40b>
 3c74319:      	movb	$0x1, 0x4(%rbx)
 3c7431d:      	jmp	0x3c73edb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x40b>
 3c74322:      	movq	%rbx, %rdi
 3c74325:      	callq	*0x14ccd45(%rip)        # 0x5141070 <writev+0x5141070>
 3c7432b:      	jmp	0x3c74129 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x659>
 3c74330:      	callq	*0x14ccd42(%rip)        # 0x5141078 <writev+0x5141078>
 3c74336:      	testb	%al, %al
 3c74338:      	jne	0x3c7411c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x64c>
 3c7433e:      	movb	$0x1, 0x4(%rbx)
 3c74342:      	jmp	0x3c7411c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x64c>
 3c74347:      	movq	%rax, %r15
 3c7434a:      	movq	0xc8(%rsp), %rax
 3c74352:      	movq	%rax, 0x60(%r14)
 3c74356:      	movups	0xb8(%rsp), %xmm0
 3c7435e:      	movups	%xmm0, 0x50(%r14)
 3c74363:      	movups	0xa8(%rsp), %xmm0
 3c7436b:      	movups	%xmm0, 0x40(%r14)
 3c74370:      	movdqu	0x68(%rsp), %xmm0
 3c74376:      	movdqu	0x78(%rsp), %xmm1
 3c7437c:      	movdqu	0x88(%rsp), %xmm2
 3c74385:      	movdqu	0x98(%rsp), %xmm3
 3c7438e:      	movdqu	%xmm3, 0x30(%r14)
 3c74394:      	movdqu	%xmm2, 0x20(%r14)
 3c7439a:      	movdqu	%xmm1, 0x10(%r14)
 3c743a0:      	movdqu	%xmm0, (%r14)
 3c743a5:      	jmp	0x3c744a9 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9d9>
 3c743aa:      	movq	%rax, %r15
 3c743ad:      	movq	0x60(%r13), %rax
 3c743b1:      	movq	%rax, 0x60(%r14)
 3c743b5:      	movups	0x50(%r13), %xmm0
 3c743ba:      	movups	%xmm0, 0x50(%r14)
 3c743bf:      	movups	0x40(%r13), %xmm0
 3c743c4:      	movups	%xmm0, 0x40(%r14)
 3c743c9:      	movdqu	(%r13), %xmm0
 3c743cf:      	movdqu	0x10(%r13), %xmm1
 3c743d5:      	movdqu	0x20(%r13), %xmm2
 3c743db:      	movdqu	0x30(%r13), %xmm3
 3c743e1:      	movdqu	%xmm3, 0x30(%r14)
 3c743e7:      	movdqu	%xmm2, 0x20(%r14)
 3c743ed:      	movdqu	%xmm1, 0x10(%r14)
 3c743f3:      	movdqu	%xmm0, (%r14)
 3c743f8:      	xorl	%ebp, %ebp
 3c743fa:      	jmp	0x3c744c8 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9f8>
 3c743ff:      	movq	%rax, %rdi
 3c74402:      	callq	*0x14cca00(%rip)        # 0x5140e08 <writev+0x5140e08>
 3c74408:      	movq	%rax, 0x58(%rsp)
 3c7440d:      	movq	%rdx, 0x60(%rsp)
 3c74412:      	movq	$-0x2, 0x50(%rsp)
 3c7441b:      	movq	0x14ccc46(%rip), %r15   # 0x5141068 <writev+0x5141068>
 3c74422:      	movq	0x8(%rsp), %r12
 3c74427:      	movl	$0x1, %ecx
 3c7442c:      	xorl	%eax, %eax
 3c7442e:      	lock
 3c7442f:      	cmpxchgl	%ecx, (%rbx)
 3c74432:      	je	0x3c73f2a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x45a>
 3c74438:      	movq	%rbx, %rdi
 3c7443b:      	callq	*0x14ccfc7(%rip)        # 0x5141408 <writev+0x5141408>
 3c74441:      	jmp	0x3c73f2a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x45a>
 3c74446:      	callq	*0x14cc9d4(%rip)        # 0x5140e20 <writev+0x5140e20>
 3c7444c:      	movq	%rax, %r15
 3c7444f:      	jmp	0x3c744b6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9e6>
 3c74451:      	movq	%rax, %r15
 3c74454:      	jmp	0x3c744dc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa0c>
 3c74459:      	movq	%rax, %r15
 3c7445c:      	movq	(%r13), %rsi
 3c74460:      	testq	%rsi, %rsi
 3c74463:      	jle	0x3c744e4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa14>
 3c74465:      	movq	0x8(%r13), %rdi
 3c74469:      	movl	$0x1, %edx
 3c7446e:      	callq	*0x14cc8cc(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c74474:      	movq	%r15, %rdi
 3c74477:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c7447c:      	movq	%rax, %r15
 3c7447f:      	leaq	0x68(%rsp), %rdi
 3c74484:      	callq	0x3c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c74489:      	jmp	0x3c744b6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0x9e6>
 3c7448b:      	callq	*0x14cc927(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c74491:      	movq	%rax, %r15
 3c74494:      	leaq	0x68(%rsp), %rdi
 3c74499:      	callq	0x3c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c7449e:      	jmp	0x3c744dc <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa0c>
 3c744a0:      	callq	*0x14cc912(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c744a6:      	movq	%rax, %r15
 3c744a9:      	movzbl	0x18(%rsp), %esi
 3c744ae:      	movq	%rbx, %rdi
 3c744b1:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c744b6:      	leaq	0x50(%rsp), %rdi
 3c744bb:      	callq	0x3bfff70 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtB4_6result6ResultIBC_uNtNtCscdodAO9FK5_5alloc6string6StringEINtNtB16_5boxed3BoxDNtNtB4_3any3AnyNtNtB4_6marker4SendEL_EEECs3pwlnhBXFtN_12memra_server>
 3c744c0:      	jmp	0x3c744e4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa14>
 3c744c2:      	movq	%rax, %r15
 3c744c5:      	movb	$0x1, %bpl
 3c744c8:      	movl	0x20(%rsp), %eax
 3c744cc:      	movzbl	%al, %esi
 3c744cf:      	movq	%rbx, %rdi
 3c744d2:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c744d7:      	testb	%bpl, %bpl
 3c744da:      	je	0x3c744e4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore16publish_terminal+0xa14>
 3c744dc:      	movq	%r13, %rdi
 3c744df:      	callq	0x3c08980 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c744e4:      	movq	%r15, %rdi
 3c744e7:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c744ec:      	callq	*0x14cc8c6(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c744f2:      	int3
 3c744f3:      	int3
 3c744f4:      	int3
 3c744f5:      	int3
 3c744f6:      	int3
 3c744f7:      	int3
 3c744f8:      	int3
 3c744f9:      	int3
 3c744fa:      	int3
 3c744fb:      	int3
 3c744fc:      	int3
 3c744fd:      	int3
 3c744fe:      	int3
 3c744ff:      	int3

0000000003c74500 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get>:
 3c74500:      	pushq	%rbp
 3c74501:      	pushq	%r15
 3c74503:      	pushq	%r14
 3c74505:      	pushq	%r13
 3c74507:      	pushq	%r12
 3c74509:      	pushq	%rbx
 3c7450a:      	subq	$0x108, %rsp            # imm = 0x108
 3c74511:      	movq	%rcx, %r13
 3c74514:      	movq	%rdx, %r12
 3c74517:      	movq	%rsi, %rbx
 3c7451a:      	movq	%rdi, %r14
 3c7451d:      	movl	$0x1, %ecx
 3c74522:      	xorl	%eax, %eax
 3c74524:      	lock
 3c74525:      	cmpxchgl	%ecx, (%rsi)
 3c74528:      	jne	0x3c74919 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x419>
 3c7452e:      	movq	0x14ccb33(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c74535:      	movq	(%rax), %rax
 3c74538:      	shlq	%rax
 3c7453b:      	testq	%rax, %rax
 3c7453e:      	jne	0x3c74927 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x427>
 3c74544:      	xorl	%r15d, %r15d
 3c74547:      	movzbl	0x4(%rbx), %eax
 3c7454b:      	testb	%al, %al
 3c7454d:      	jne	0x3c74940 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x440>
 3c74553:      	leaq	0x8(%rbx), %rsi
 3c74557:      	movq	%rbx, %rdi
 3c7455a:      	callq	0x3c47610 <_RNvMNtCs3pwlnhBXFtN_12memra_server9job_storeNtB2_16InMemoryJobStore12sweep_locked.llvm.12140772379724168186>
 3c7455f:      	cmpq	$0x0, 0x20(%rbx)
 3c74564:      	je	0x3c746a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x1a1>
 3c7456a:      	movl	%r15d, 0xc(%rsp)
 3c7456f:      	leaq	0x28(%rbx), %rdi
 3c74573:      	movq	%r12, %rsi
 3c74576:      	movq	%r13, %rdx
 3c74579:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c7457e:      	movq	%rax, %rbp
 3c74581:      	shrq	$0x39, %rax
 3c74585:      	movq	0x8(%rbx), %rcx
 3c74589:      	movq	0x10(%rbx), %rdx
 3c7458d:      	movd	%eax, %xmm0
 3c74591:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74595:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c7459a:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c7459f:      	xorl	%esi, %esi
 3c745a1:      	pcmpeqd	%xmm2, %xmm2
 3c745a5:      	movq	0x14cc7fc(%rip), %r8    # 0x5140da8 <writev+0x5140da8>
 3c745ac:      	andq	%rdx, %rbp
 3c745af:      	movdqu	(%rcx,%rbp), %xmm3
 3c745b4:      	movdqa	%xmm3, %xmm0
 3c745b8:      	pcmpeqb	%xmm1, %xmm0
 3c745bc:      	pmovmskb	%xmm0, %r15d
 3c745c1:      	testl	%r15d, %r15d
 3c745c4:      	je	0x3c74680 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x180>
 3c745ca:      	movq	%r12, 0x18(%rsp)
 3c745cf:      	movq	%rcx, 0x78(%rsp)
 3c745d4:      	movq	%rdx, 0x70(%rsp)
 3c745d9:      	movdqa	%xmm1, 0xe0(%rsp)
 3c745e2:      	movq	%rsi, 0x68(%rsp)
 3c745e7:      	movdqa	%xmm3, 0xd0(%rsp)
 3c745f0:      	tzcntl	%r15d, %eax
 3c745f5:      	addq	%rbp, %rax
 3c745f8:      	andq	%rdx, %rax
 3c745fb:      	shlq	$0x8, %rax
 3c745ff:      	movq	%r13, %r12
 3c74602:      	movq	%rcx, %r13
 3c74605:      	subq	%rax, %r13
 3c74608:      	cmpq	-0xf0(%r13), %r12
 3c7460f:      	jne	0x3c7463d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x13d>
 3c74611:      	movq	-0xf8(%r13), %rsi
 3c74618:      	movq	0x18(%rsp), %rdi
 3c7461d:      	movq	%r12, %rdx
 3c74620:      	movq	%rbx, 0x10(%rsp)
 3c74625:      	movq	%r14, %rbx
 3c74628:      	movq	%r8, %r14
 3c7462b:      	callq	*%r8
 3c7462e:      	movq	%r14, %r8
 3c74631:      	movq	%rbx, %r14
 3c74634:      	movq	0x10(%rsp), %rbx
 3c74639:      	testl	%eax, %eax
 3c7463b:      	je	0x3c746b6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x1b6>
 3c7463d:      	leal	-0x1(%r15), %eax
 3c74641:      	andw	%r15w, %ax
 3c74645:      	movl	%eax, %r15d
 3c74648:      	movq	%r12, %r13
 3c7464b:      	movq	0x18(%rsp), %r12
 3c74650:      	movq	0x78(%rsp), %rcx
 3c74655:      	movq	0x70(%rsp), %rdx
 3c7465a:      	movdqa	0xe0(%rsp), %xmm1
 3c74663:      	movq	0x68(%rsp), %rsi
 3c74668:      	pcmpeqd	%xmm2, %xmm2
 3c7466c:      	movdqa	0xd0(%rsp), %xmm3
 3c74675:      	jne	0x3c745f0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0xf0>
 3c7467b:      	nopl	(%rax,%rax)
 3c74680:      	pcmpeqb	%xmm2, %xmm3
 3c74684:      	pmovmskb	%xmm3, %eax
 3c74688:      	testl	%eax, %eax
 3c7468a:      	movl	0xc(%rsp), %r15d
 3c7468f:      	jne	0x3c746a1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x1a1>
 3c74691:      	addq	%rsi, %rbp
 3c74694:      	addq	$0x10, %rbp
 3c74698:      	addq	$0x10, %rsi
 3c7469c:      	jmp	0x3c745ac <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0xac>
 3c746a1:      	movq	$-0x2, (%r14)
 3c746a8:      	testb	%r15b, %r15b
 3c746ab:      	je	0x3c748de <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3de>
 3c746b1:      	jmp	0x3c748fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c746b6:      	movzbl	-0x88(%r13), %ebp
 3c746be:      	movq	-0xd0(%r13), %rcx
 3c746c5:      	cmpq	$-0x1, %rcx
 3c746c9:      	je	0x3c74710 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x210>
 3c746cb:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c746d5:      	incq	%rax
 3c746d8:      	movq	%rcx, %rdx
 3c746db:      	xorq	%rax, %rdx
 3c746de:      	testq	%rcx, %rcx
 3c746e1:      	movl	$0x5, %ecx
 3c746e6:      	cmovsq	%rdx, %rcx
 3c746ea:      	leaq	-0xd0(%r13), %rsi
 3c746f1:      	leaq	-0x2e56204(%rip), %rdx  # 0xe1e4f4 <anon.579209908c49646e156899b92400b4a2.287.llvm.9148498120539102879+0x148d>
 3c746f8:      	movslq	(%rdx,%rcx,4), %rcx
 3c746fc:      	addq	%rdx, %rcx
 3c746ff:      	movl	0xc(%rsp), %r15d
 3c74704:      	jmpq	*%rcx
 3c74706:      	movq	%rax, 0x20(%rsp)
 3c7470b:      	jmp	0x3c747fd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c74710:      	movq	$-0x1, 0x80(%rsp)
 3c7471c:      	movl	0xc(%rsp), %r15d
 3c74721:      	addq	$-0xe8, %r13
 3c74728:      	cmpq	$-0x1, (%r13)
 3c7472d:      	jne	0x3c74863 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x363>
 3c74733:      	movq	$-0x1, %rax
 3c7473a:      	jmp	0x3c74885 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x385>
 3c7473f:      	leaq	0x28(%rsp), %rdi
 3c74744:      	movq	-0xc0(%r13), %rsi
 3c7474b:      	movq	-0xb8(%r13), %rdx
 3c74752:      	callq	0x3afa8f0 <_RINvXNvMNtCscdodAO9FK5_5alloc5sliceSp9to_vec_inNtNtCs8OSp0AlFmbY_10serde_json5value5ValueNtB3_10ConvertVec6to_vecNtNtB8_5alloc6GlobalECs3pwlnhBXFtN_12memra_server.llvm.7562590961585166920>
 3c74757:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c74761:      	addq	$0x5, %rax
 3c74765:      	movq	%rax, 0x20(%rsp)
 3c7476a:      	jmp	0x3c747fd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c7476f:      	leaq	-0xc8(%r13), %rsi
 3c74776:      	leaq	0x28(%rsp), %rdi
 3c7477b:      	callq	*0x14cc817(%rip)        # 0x5140f98 <writev+0x5140f98>
 3c74781:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c7478b:      	addq	$0x3, %rax
 3c7478f:      	movq	%rax, 0x20(%rsp)
 3c74794:      	jmp	0x3c747fd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c74796:      	leaq	-0xc8(%r13), %rsi
 3c7479d:      	leaq	0x28(%rsp), %rdi
 3c747a2:      	callq	*0x14cc7f0(%rip)        # 0x5140f98 <writev+0x5140f98>
 3c747a8:      	movabsq	$0x7fffffffffffffff, %rax # imm = 0x7FFFFFFFFFFFFFFF
 3c747b2:      	addq	$0x4, %rax
 3c747b6:      	movq	%rax, 0x20(%rsp)
 3c747bb:      	jmp	0x3c747fd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c747bd:      	movq	0x40(%rsi), %rax
 3c747c1:      	movq	%rax, 0x60(%rsp)
 3c747c6:      	movdqu	(%rsi), %xmm0
 3c747ca:      	movdqu	0x10(%rsi), %xmm1
 3c747cf:      	movdqu	0x20(%rsi), %xmm2
 3c747d4:      	movdqu	0x30(%rsi), %xmm3
 3c747d9:      	movdqa	%xmm3, 0x50(%rsp)
 3c747df:      	movdqa	%xmm2, 0x40(%rsp)
 3c747e5:      	movdqa	%xmm1, 0x30(%rsp)
 3c747eb:      	movdqa	%xmm0, 0x20(%rsp)
 3c747f1:      	jmp	0x3c747fd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2fd>
 3c747f3:      	leaq	0x20(%rsp), %rdi
 3c747f8:      	callq	0x3a01480 <_RNvXNtCseAOs6zoTI9e_8indexmap3mapINtB2_8IndexMapNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtCs4NRVxsYgnAr_4core5clone5Clone5cloneCs3pwlnhBXFtN_12memra_server>
 3c747fd:      	movq	0x60(%rsp), %rax
 3c74802:      	movq	%rax, 0xc0(%rsp)
 3c7480a:      	movq	0x20(%rsp), %rax
 3c7480f:      	movq	0x28(%rsp), %rcx
 3c74814:      	movdqa	0x30(%rsp), %xmm0
 3c7481a:      	movdqa	0x40(%rsp), %xmm1
 3c74820:      	movdqa	0x50(%rsp), %xmm2
 3c74826:      	movdqa	%xmm2, 0xb0(%rsp)
 3c7482f:      	movdqa	%xmm1, 0xa0(%rsp)
 3c74838:      	movdqa	%xmm0, 0x90(%rsp)
 3c74841:      	movq	%rax, 0x80(%rsp)
 3c74849:      	movq	%rcx, 0x88(%rsp)
 3c74851:      	addq	$-0xe8, %r13
 3c74858:      	cmpq	$-0x1, (%r13)
 3c7485d:      	je	0x3c74733 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x233>
 3c74863:      	leaq	0x20(%rsp), %rdi
 3c74868:      	movq	%r13, %rsi
 3c7486b:      	callq	*0x14cc727(%rip)        # 0x5140f98 <writev+0x5140f98>
 3c74871:      	movq	0x20(%rsp), %rax
 3c74876:      	movdqu	0x28(%rsp), %xmm0
 3c7487c:      	movdqa	%xmm0, 0xf0(%rsp)
 3c74885:      	movq	0xc0(%rsp), %rcx
 3c7488d:      	movq	%rcx, 0x58(%r14)
 3c74891:      	movaps	0x80(%rsp), %xmm0
 3c74899:      	movaps	0x90(%rsp), %xmm1
 3c748a1:      	movaps	0xa0(%rsp), %xmm2
 3c748a9:      	movaps	0xb0(%rsp), %xmm3
 3c748b1:      	movups	%xmm3, 0x48(%r14)
 3c748b6:      	movups	%xmm2, 0x38(%r14)
 3c748bb:      	movups	%xmm1, 0x28(%r14)
 3c748c0:      	movups	%xmm0, 0x18(%r14)
 3c748c5:      	movq	%rax, (%r14)
 3c748c8:      	movaps	0xf0(%rsp), %xmm0
 3c748d0:      	movups	%xmm0, 0x8(%r14)
 3c748d5:      	movb	%bpl, 0x60(%r14)
 3c748d9:      	testb	%r15b, %r15b
 3c748dc:      	jne	0x3c748fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c748de:      	movq	0x14cc783(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c748e5:      	movq	(%rax), %rax
 3c748e8:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c748f2:      	testq	%rcx, %rax
 3c748f5:      	jne	0x3c7497c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x47c>
 3c748fb:      	xorl	%eax, %eax
 3c748fd:      	xchgl	%eax, (%rbx)
 3c748ff:      	cmpl	$0x2, %eax
 3c74902:      	je	0x3c74971 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x471>
 3c74904:      	movq	%r14, %rax
 3c74907:      	addq	$0x108, %rsp            # imm = 0x108
 3c7490e:      	popq	%rbx
 3c7490f:      	popq	%r12
 3c74911:      	popq	%r13
 3c74913:      	popq	%r14
 3c74915:      	popq	%r15
 3c74917:      	popq	%rbp
 3c74918:      	retq
 3c74919:      	movq	%rbx, %rdi
 3c7491c:      	callq	*0x14ccae6(%rip)        # 0x5141408 <writev+0x5141408>
 3c74922:      	jmp	0x3c7452e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x2e>
 3c74927:      	callq	*0x14cc74b(%rip)        # 0x5141078 <writev+0x5141078>
 3c7492d:      	movl	%eax, %r15d
 3c74930:      	xorb	$0x1, %r15b
 3c74934:      	movzbl	0x4(%rbx), %eax
 3c74938:      	testb	%al, %al
 3c7493a:      	je	0x3c74553 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x53>
 3c74940:      	movq	%rbx, 0x20(%rsp)
 3c74945:      	movb	%r15b, 0x28(%rsp)
 3c7494a:      	leaq	-0x2e55f28(%rip), %rdi  # 0xe1ea29 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c74951:      	leaq	0x13b7b98(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c74958:      	leaq	0x13b8141(%rip), %r8    # 0x502caa0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2e0>
 3c7495f:      	leaq	0x20(%rsp), %rdx
 3c74964:      	movl	$0x2b, %esi
 3c74969:      	callq	*0x14cc529(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c7496f:      	ud2
 3c74971:      	movq	%rbx, %rdi
 3c74974:      	callq	*0x14cc6f6(%rip)        # 0x5141070 <writev+0x5141070>
 3c7497a:      	jmp	0x3c74904 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x404>
 3c7497c:      	callq	*0x14cc6f6(%rip)        # 0x5141078 <writev+0x5141078>
 3c74982:      	testb	%al, %al
 3c74984:      	jne	0x3c748fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c7498a:      	movb	$0x1, 0x4(%rbx)
 3c7498e:      	jmp	0x3c748fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x3fb>
 3c74993:      	movq	%rax, %r14
 3c74996:      	cmpq	$-0x1, 0x80(%rsp)
 3c7499f:      	je	0x3c749c3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x4c3>
 3c749a1:      	leaq	0x80(%rsp), %rdi
 3c749a9:      	callq	0x3c08af0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c749ae:      	jmp	0x3c749c3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x4c3>
 3c749b0:      	callq	*0x14cc402(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c749b6:      	movq	%rbx, 0x10(%rsp)
 3c749bb:      	movl	%r15d, 0xc(%rsp)
 3c749c0:      	movq	%rax, %r14
 3c749c3:      	movzbl	0xc(%rsp), %esi
 3c749c8:      	movq	0x10(%rsp), %rdi
 3c749cd:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c749d2:      	jmp	0x3c749e7 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3get+0x4e7>
 3c749d4:      	callq	*0x14cc3de(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c749da:      	movq	%rax, %r14
 3c749dd:      	leaq	0x20(%rsp), %rdi
 3c749e2:      	callq	0x3c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c749e7:      	movq	%r14, %rdi
 3c749ea:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c749ef:      	callq	*0x14cc3c3(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c749f5:      	int3
 3c749f6:      	int3
 3c749f7:      	int3
 3c749f8:      	int3
 3c749f9:      	int3
 3c749fa:      	int3
 3c749fb:      	int3
 3c749fc:      	int3
 3c749fd:      	int3
 3c749fe:      	int3
 3c749ff:      	int3

0000000003c74a00 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put>:
 3c74a00:      	pushq	%rbp
 3c74a01:      	pushq	%r15
 3c74a03:      	pushq	%r14
 3c74a05:      	pushq	%r13
 3c74a07:      	pushq	%r12
 3c74a09:      	pushq	%rbx
 3c74a0a:      	subq	$0x298, %rsp            # imm = 0x298
 3c74a11:      	movq	%rcx, %r13
 3c74a14:      	movq	%rdx, %r12
 3c74a17:      	movq	%rsi, 0x18(%rsp)
 3c74a1c:      	movq	%rdi, %r14
 3c74a1f:      	movl	$0x1, %ecx
 3c74a24:      	xorl	%eax, %eax
 3c74a26:      	lock
 3c74a27:      	cmpxchgl	%ecx, (%rdi)
 3c74a2a:      	jne	0x3c75113 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x713>
 3c74a30:      	movq	0x14cc631(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c74a37:      	movq	(%rax), %rax
 3c74a3a:      	shlq	%rax
 3c74a3d:      	testq	%rax, %rax
 3c74a40:      	jne	0x3c75121 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x721>
 3c74a46:      	xorl	%eax, %eax
 3c74a48:      	movzbl	0x4(%r14), %ecx
 3c74a4d:      	testb	%cl, %cl
 3c74a4f:      	jne	0x3c75136 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x736>
 3c74a55:      	movl	%eax, 0x24(%rsp)
 3c74a59:      	leaq	0x8(%r14), %rsi
 3c74a5d:      	movb	$0x1, %al
 3c74a5f:      	movl	%eax, 0x4(%rsp)
 3c74a63:      	movq	%r14, %rdi
 3c74a66:      	callq	0x3c47610 <_RNvMNtCs3pwlnhBXFtN_12memra_server9job_storeNtB2_16InMemoryJobStore12sweep_locked.llvm.12140772379724168186>
 3c74a6b:      	cmpq	$0x0, 0x20(%r14)
 3c74a70:      	movq	%r12, 0x10(%rsp)
 3c74a75:      	je	0x3c74b81 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x181>
 3c74a7b:      	leaq	0x28(%r14), %rdi
 3c74a7f:      	movq	0x18(%rsp), %rsi
 3c74a84:      	movq	%r12, %rdx
 3c74a87:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c74a8c:      	movq	%rax, %rcx
 3c74a8f:      	shrq	$0x39, %rcx
 3c74a93:      	movq	0x8(%r14), %r15
 3c74a97:      	movq	0x10(%r14), %rbp
 3c74a9b:      	movd	%ecx, %xmm0
 3c74a9f:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74aa3:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c74aa8:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c74aad:      	xorl	%edx, %edx
 3c74aaf:      	pcmpeqd	%xmm2, %xmm2
 3c74ab3:      	movq	0x14cc2ee(%rip), %rbx   # 0x5140da8 <writev+0x5140da8>
 3c74aba:      	andq	%rbp, %rax
 3c74abd:      	movdqu	(%r15,%rax), %xmm3
 3c74ac3:      	movdqa	%xmm3, %xmm0
 3c74ac7:      	pcmpeqb	%xmm1, %xmm0
 3c74acb:      	pmovmskb	%xmm0, %r12d
 3c74ad0:      	testl	%r12d, %r12d
 3c74ad3:      	je	0x3c74b60 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x160>
 3c74ad9:      	movq	%r13, 0x28(%rsp)
 3c74ade:      	movdqa	%xmm1, 0x30(%rsp)
 3c74ae4:      	movq	%rdx, 0x8(%rsp)
 3c74ae9:      	movq	%rax, 0x60(%rsp)
 3c74aee:      	movdqa	%xmm3, 0x50(%rsp)
 3c74af4:      	tzcntl	%r12d, %ecx
 3c74af9:      	addq	%rax, %rcx
 3c74afc:      	andq	%rbp, %rcx
 3c74aff:      	shlq	$0x8, %rcx
 3c74b03:      	movq	%r15, %r13
 3c74b06:      	subq	%rcx, %r13
 3c74b09:      	movq	0x10(%rsp), %rdx
 3c74b0e:      	cmpq	-0xf0(%r13), %rdx
 3c74b15:      	jne	0x3c74b29 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x129>
 3c74b17:      	movq	-0xf8(%r13), %rsi
 3c74b1e:      	movq	0x18(%rsp), %rdi
 3c74b23:      	callq	*%rbx
 3c74b25:      	testl	%eax, %eax
 3c74b27:      	je	0x3c74b9a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x19a>
 3c74b29:      	leal	-0x1(%r12), %eax
 3c74b2e:      	andw	%r12w, %ax
 3c74b32:      	movl	%eax, %r12d
 3c74b35:      	movq	0x60(%rsp), %rax
 3c74b3a:      	movq	0x28(%rsp), %r13
 3c74b3f:      	movdqa	0x30(%rsp), %xmm1
 3c74b45:      	movq	0x8(%rsp), %rdx
 3c74b4a:      	pcmpeqd	%xmm2, %xmm2
 3c74b4e:      	movdqa	0x50(%rsp), %xmm3
 3c74b54:      	jne	0x3c74af4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0xf4>
 3c74b56:      	nopw	%cs:(%rax,%rax)
 3c74b60:      	pcmpeqb	%xmm2, %xmm3
 3c74b64:      	pmovmskb	%xmm3, %ecx
 3c74b68:      	testl	%ecx, %ecx
 3c74b6a:      	movq	0x10(%rsp), %r12
 3c74b6f:      	jne	0x3c74b81 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x181>
 3c74b71:      	addq	%rdx, %rax
 3c74b74:      	addq	$0x10, %rax
 3c74b78:      	addq	$0x10, %rdx
 3c74b7c:      	jmp	0x3c74aba <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0xba>
 3c74b81:      	cmpb	$0x0, 0x60(%r13)
 3c74b86:      	je	0x3c74bd0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x1d0>
 3c74b88:      	xorl	%ebx, %ebx
 3c74b8a:      	cmpb	$0x0, 0x24(%rsp)
 3c74b8f:      	je	0x3c74ea9 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4a9>
 3c74b95:      	jmp	0x3c74ec6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c74b9a:      	movb	$0x1, %bl
 3c74b9c:      	cmpb	$0x1, -0x88(%r13)
 3c74ba4:      	jbe	0x3c74bbb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x1bb>
 3c74ba6:      	movq	0x28(%rsp), %r13
 3c74bab:      	cmpb	$0x0, 0x24(%rsp)
 3c74bb0:      	je	0x3c74ea9 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4a9>
 3c74bb6:      	jmp	0x3c74ec6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c74bbb:      	cmpq	$-0x2, -0x80(%r13)
 3c74bc0:      	movq	0x28(%rsp), %r13
 3c74bc5:      	movq	0x10(%rsp), %r12
 3c74bca:      	jne	0x3c74ea2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4a2>
 3c74bd0:      	movq	%r13, %rdi
 3c74bd3:      	callq	0x3c6efc0 <_RNvNtCs3pwlnhBXFtN_12memra_server9job_store17record_size_bytes>
 3c74bd8:      	movq	%rax, %r15
 3c74bdb:      	cmpb	$0x1, 0x60(%r13)
 3c74be0:      	movq	%r13, 0x28(%rsp)
 3c74be5:      	jbe	0x3c74c09 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x209>
 3c74be7:      	callq	*0x14cc2c3(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3c74bed:      	movq	%rax, 0x60(%rsp)
 3c74bf2:      	cmpq	$0x0, 0x20(%r14)
 3c74bf7:      	jne	0x3c74d50 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x350>
 3c74bfd:      	movl	%edx, 0x30(%rsp)
 3c74c01:      	xorl	%r13d, %r13d
 3c74c04:      	jmp	0x3c74e7f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47f>
 3c74c09:      	cmpq	$0x0, 0x20(%r14)
 3c74c0e:      	je	0x3c74d3b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x33b>
 3c74c14:      	leaq	0x28(%r14), %rdi
 3c74c18:      	movq	0x18(%rsp), %rsi
 3c74c1d:      	movq	%r12, %rdx
 3c74c20:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c74c25:      	movq	%rax, %rcx
 3c74c28:      	shrq	$0x39, %rcx
 3c74c2c:      	movq	0x8(%r14), %r13
 3c74c30:      	movq	0x10(%r14), %rdi
 3c74c34:      	movd	%ecx, %xmm0
 3c74c38:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74c3c:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c74c41:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c74c46:      	xorl	%esi, %esi
 3c74c48:      	pcmpeqd	%xmm2, %xmm2
 3c74c4c:      	movq	0x14cc155(%rip), %rbp   # 0x5140da8 <writev+0x5140da8>
 3c74c53:      	andq	%rdi, %rax
 3c74c56:      	movdqu	(%r13,%rax), %xmm3
 3c74c5d:      	movdqa	%xmm3, %xmm0
 3c74c61:      	pcmpeqb	%xmm1, %xmm0
 3c74c65:      	pmovmskb	%xmm0, %r12d
 3c74c6a:      	testl	%r12d, %r12d
 3c74c6d:      	je	0x3c74cfd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x2fd>
 3c74c73:      	movq	%rax, 0x30(%rsp)
 3c74c78:      	movq	%r15, 0x8(%rsp)
 3c74c7d:      	movdqa	%xmm1, 0x60(%rsp)
 3c74c83:      	movq	%rsi, 0x50(%rsp)
 3c74c88:      	movq	%rdi, 0x48(%rsp)
 3c74c8d:      	movdqa	%xmm3, 0x70(%rsp)
 3c74c93:      	tzcntl	%r12d, %ecx
 3c74c98:      	addq	%rax, %rcx
 3c74c9b:      	andq	%rdi, %rcx
 3c74c9e:      	shlq	$0x8, %rcx
 3c74ca2:      	movq	%r13, %rbx
 3c74ca5:      	subq	%rcx, %rbx
 3c74ca8:      	movq	0x10(%rsp), %r15
 3c74cad:      	cmpq	-0xf0(%rbx), %r15
 3c74cb4:      	jne	0x3c74ccb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x2cb>
 3c74cb6:      	movq	-0xf8(%rbx), %rsi
 3c74cbd:      	movq	0x18(%rsp), %rdi
 3c74cc2:      	movq	%r15, %rdx
 3c74cc5:      	callq	*%rbp
 3c74cc7:      	testl	%eax, %eax
 3c74cc9:      	je	0x3c74d1e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x31e>
 3c74ccb:      	leal	-0x1(%r12), %eax
 3c74cd0:      	andw	%r12w, %ax
 3c74cd4:      	movl	%eax, %r12d
 3c74cd7:      	movq	0x30(%rsp), %rax
 3c74cdc:      	movq	0x8(%rsp), %r15
 3c74ce1:      	movdqa	0x60(%rsp), %xmm1
 3c74ce7:      	movq	0x50(%rsp), %rsi
 3c74cec:      	pcmpeqd	%xmm2, %xmm2
 3c74cf0:      	movq	0x48(%rsp), %rdi
 3c74cf5:      	movdqa	0x70(%rsp), %xmm3
 3c74cfb:      	jne	0x3c74c93 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x293>
 3c74cfd:      	pcmpeqb	%xmm2, %xmm3
 3c74d01:      	pmovmskb	%xmm3, %ecx
 3c74d05:      	movl	$0xffffffff, %edx       # imm = 0xFFFFFFFF
 3c74d0a:      	testl	%ecx, %ecx
 3c74d0c:      	jne	0x3c74d4b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x34b>
 3c74d0e:      	addq	%rsi, %rax
 3c74d11:      	addq	$0x10, %rax
 3c74d15:      	addq	$0x10, %rsi
 3c74d19:      	jmp	0x3c74c53 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x253>
 3c74d1e:      	movq	-0x8(%rbx), %rax
 3c74d22:      	movq	0x8(%rsp), %rcx
 3c74d27:      	cmpq	%rcx, %rax
 3c74d2a:      	cmovaq	%rax, %rcx
 3c74d2e:      	movl	$0xffffffff, %edx       # imm = 0xFFFFFFFF
 3c74d33:      	movq	%r15, %r12
 3c74d36:      	movq	%rcx, %r15
 3c74d39:      	jmp	0x3c74d50 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x350>
 3c74d3b:      	movl	$0xffffffff, 0x30(%rsp) # imm = 0xFFFFFFFF
 3c74d43:      	xorl	%r13d, %r13d
 3c74d46:      	jmp	0x3c74e7f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47f>
 3c74d4b:      	movq	0x10(%rsp), %r12
 3c74d50:      	movl	%edx, 0x30(%rsp)
 3c74d54:      	leaq	0x28(%r14), %rdi
 3c74d58:      	movq	0x18(%rsp), %rsi
 3c74d5d:      	movq	%r12, %rdx
 3c74d60:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c74d65:      	movq	%rax, %rcx
 3c74d68:      	shrq	$0x39, %rcx
 3c74d6c:      	movq	0x8(%r14), %rsi
 3c74d70:      	movq	0x10(%r14), %rdx
 3c74d74:      	movd	%ecx, %xmm0
 3c74d78:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c74d7c:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c74d81:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c74d86:      	xorl	%r13d, %r13d
 3c74d89:      	pcmpeqd	%xmm2, %xmm2
 3c74d8d:      	movq	0x14cc014(%rip), %rbp   # 0x5140da8 <writev+0x5140da8>
 3c74d94:      	xorl	%edi, %edi
 3c74d96:      	movq	0x10(%rsp), %r12
 3c74d9b:      	andq	%rdx, %rax
 3c74d9e:      	movdqu	(%rsi,%rax), %xmm3
 3c74da3:      	movdqa	%xmm3, %xmm0
 3c74da7:      	pcmpeqb	%xmm1, %xmm0
 3c74dab:      	pmovmskb	%xmm0, %ebx
 3c74daf:      	testl	%ebx, %ebx
 3c74db1:      	je	0x3c74e53 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x453>
 3c74db7:      	movq	%rax, 0x50(%rsp)
 3c74dbc:      	movq	%r15, 0x8(%rsp)
 3c74dc1:      	movq	%rdx, 0x48(%rsp)
 3c74dc6:      	movdqa	%xmm1, 0x70(%rsp)
 3c74dcc:      	movq	%rsi, 0x90(%rsp)
 3c74dd4:      	movq	%rdi, 0x88(%rsp)
 3c74ddc:      	movdqa	%xmm3, 0xb0(%rsp)
 3c74de5:      	tzcntl	%ebx, %ecx
 3c74de9:      	addq	%rax, %rcx
 3c74dec:      	andq	%rdx, %rcx
 3c74def:      	shlq	$0x8, %rcx
 3c74df3:      	movq	%rsi, %r15
 3c74df6:      	subq	%rcx, %r15
 3c74df9:      	cmpq	-0xf0(%r15), %r12
 3c74e00:      	jne	0x3c74e17 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x417>
 3c74e02:      	movq	-0xf8(%r15), %rsi
 3c74e09:      	movq	0x18(%rsp), %rdi
 3c74e0e:      	movq	%r12, %rdx
 3c74e11:      	callq	*%rbp
 3c74e13:      	testl	%eax, %eax
 3c74e15:      	je	0x3c74e6f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x46f>
 3c74e17:      	leal	-0x1(%rbx), %eax
 3c74e1a:      	andw	%bx, %ax
 3c74e1d:      	movl	%eax, %ebx
 3c74e1f:      	movq	0x50(%rsp), %rax
 3c74e24:      	movq	0x8(%rsp), %r15
 3c74e29:      	movq	0x48(%rsp), %rdx
 3c74e2e:      	movdqa	0x70(%rsp), %xmm1
 3c74e34:      	pcmpeqd	%xmm2, %xmm2
 3c74e38:      	movq	0x90(%rsp), %rsi
 3c74e40:      	movq	0x88(%rsp), %rdi
 3c74e48:      	movdqa	0xb0(%rsp), %xmm3
 3c74e51:      	jne	0x3c74de5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x3e5>
 3c74e53:      	pcmpeqb	%xmm2, %xmm3
 3c74e57:      	pmovmskb	%xmm3, %ecx
 3c74e5b:      	testl	%ecx, %ecx
 3c74e5d:      	jne	0x3c74e7a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47a>
 3c74e5f:      	addq	%rdi, %rax
 3c74e62:      	addq	$0x10, %rax
 3c74e66:      	addq	$0x10, %rdi
 3c74e6a:      	jmp	0x3c74d9b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x39b>
 3c74e6f:      	movq	-0x8(%r15), %r13
 3c74e73:      	movq	0x8(%rsp), %r15
 3c74e78:      	jmp	0x3c74e7f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x47f>
 3c74e7a:      	movq	0x10(%rsp), %r12
 3c74e7f:      	movq	0x38(%r14), %rcx
 3c74e83:      	xorl	%eax, %eax
 3c74e85:      	subq	%r13, %rcx
 3c74e88:      	cmovaeq	%rcx, %rax
 3c74e8c:      	movb	$0x2, %bl
 3c74e8e:      	addq	%r15, %rax
 3c74e91:      	jb	0x3c74ba6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x1a6>
 3c74e97:      	cmpq	0x40(%r14), %rax
 3c74e9b:      	movq	0x28(%rsp), %r13
 3c74ea0:      	jbe	0x3c74f16 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x516>
 3c74ea2:      	cmpb	$0x0, 0x24(%rsp)
 3c74ea7:      	jne	0x3c74ec6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c74ea9:      	movq	0x14cc1b8(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c74eb0:      	movq	(%rax), %rax
 3c74eb3:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c74ebd:      	testq	%rcx, %rax
 3c74ec0:      	jne	0x3c7517d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x77d>
 3c74ec6:      	xorl	%eax, %eax
 3c74ec8:      	xchgl	%eax, (%r14)
 3c74ecb:      	cmpl	$0x2, %eax
 3c74ece:      	je	0x3c7516f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x76f>
 3c74ed4:      	cmpq	$-0x1, 0x18(%r13)
 3c74ed9:      	je	0x3c74ee4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4e4>
 3c74edb:      	leaq	0x18(%r13), %rdi
 3c74edf:      	callq	0x3c08af0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c74ee4:      	movq	(%r13), %rsi
 3c74ee8:      	cmpq	$-0x1, %rsi
 3c74eec:      	je	0x3c74f02 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c74eee:      	testq	%rsi, %rsi
 3c74ef1:      	je	0x3c74f02 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c74ef3:      	movq	0x8(%r13), %rdi
 3c74ef7:      	movl	$0x1, %edx
 3c74efc:      	callq	*0x14cbe3e(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c74f02:      	movl	%ebx, %eax
 3c74f04:      	addq	$0x298, %rsp            # imm = 0x298
 3c74f0b:      	popq	%rbx
 3c74f0c:      	popq	%r12
 3c74f0e:      	popq	%r13
 3c74f10:      	popq	%r14
 3c74f12:      	popq	%r15
 3c74f14:      	popq	%rbp
 3c74f15:      	retq
 3c74f16:      	movq	%rax, 0x38(%r14)
 3c74f1a:      	movq	%r13, %rbx
 3c74f1d:      	testq	%r12, %r12
 3c74f20:      	jns	0x3c74f38 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x538>
 3c74f22:      	xorl	%ebp, %ebp
 3c74f24:      	movq	%rbp, %rdi
 3c74f27:      	movq	%r12, %rsi
 3c74f2a:      	movq	%rbx, %r13
 3c74f2d:      	callq	*0x14cbe55(%rip)        # 0x5140d88 <writev+0x5140d88>
 3c74f33:      	jmp	0x3c7516d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x76d>
 3c74f38:      	je	0x3c74f6e <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x56e>
 3c74f3a:      	callq	*0x14cbe50(%rip)        # 0x5140d90 <writev+0x5140d90>
 3c74f40:      	movl	$0x1, %ebp
 3c74f45:      	movl	$0x1, %esi
 3c74f4a:      	movq	%r12, %rdi
 3c74f4d:      	callq	*0x14cbe45(%rip)        # 0x5140d98 <writev+0x5140d98>
 3c74f53:      	testq	%rax, %rax
 3c74f56:      	je	0x3c74f24 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x524>
 3c74f58:      	movq	%rax, %r13
 3c74f5b:      	movq	%rax, %rdi
 3c74f5e:      	movq	0x18(%rsp), %rsi
 3c74f63:      	movq	%r12, %rdx
 3c74f66:      	callq	*0x14cbdcc(%rip)        # 0x5140d38 <writev+0x5140d38>
 3c74f6c:      	jmp	0x3c74f74 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x574>
 3c74f6e:      	movl	$0x1, %r13d
 3c74f74:      	movq	%r12, 0x98(%rsp)
 3c74f7c:      	movq	%r13, 0xa0(%rsp)
 3c74f84:      	movq	%r12, 0xa8(%rsp)
 3c74f8c:      	movq	%rbx, %r13
 3c74f8f:      	movq	0x60(%rbx), %rax
 3c74f93:      	movq	%rax, 0x120(%rsp)
 3c74f9b:      	movups	0x50(%rbx), %xmm0
 3c74f9f:      	movaps	%xmm0, 0x110(%rsp)
 3c74fa7:      	movups	0x40(%rbx), %xmm0
 3c74fab:      	movaps	%xmm0, 0x100(%rsp)
 3c74fb3:      	movdqu	(%rbx), %xmm0
 3c74fb7:      	movdqu	0x10(%rbx), %xmm1
 3c74fbc:      	movdqu	0x20(%rbx), %xmm2
 3c74fc1:      	movdqu	0x30(%rbx), %xmm3
 3c74fc6:      	movdqa	%xmm3, 0xf0(%rsp)
 3c74fcf:      	movdqa	%xmm2, 0xe0(%rsp)
 3c74fd8:      	movdqa	%xmm1, 0xd0(%rsp)
 3c74fe1:      	movdqa	%xmm0, 0xc0(%rsp)
 3c74fea:      	movq	%r15, 0x1a0(%rsp)
 3c74ff2:      	movq	0x60(%rsp), %rax
 3c74ff7:      	movq	%rax, 0x190(%rsp)
 3c74fff:      	movl	0x30(%rsp), %eax
 3c75003:      	movl	%eax, 0x198(%rsp)
 3c7500a:      	movq	$-0x2, 0x128(%rsp)
 3c75016:      	movl	$0x0, 0x4(%rsp)
 3c7501e:      	leaq	0x1b0(%rsp), %rdi
 3c75026:      	leaq	0x98(%rsp), %rdx
 3c7502e:      	leaq	0xc0(%rsp), %rcx
 3c75036:      	leaq	0x8(%r14), %rsi
 3c7503a:      	callq	0x3e99340 <_RNvMs1_NtCsgQfI1edjipl_9hashbrown3mapINtB5_7HashMapNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateE6insertB1s_>
 3c7503f:      	movq	0x1b0(%rsp), %r12
 3c75047:      	cmpq	$-0x2, %r12
 3c7504b:      	je	0x3c750d1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c75051:      	cmpq	$-0x1, 0x1c8(%rsp)
 3c7505a:      	je	0x3c75069 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x669>
 3c7505c:      	leaq	0x1c8(%rsp), %rdi
 3c75064:      	callq	0x3c08af0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c75069:      	cmpq	$-0x1, %r12
 3c7506d:      	je	0x3c7508a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x68a>
 3c7506f:      	testq	%r12, %r12
 3c75072:      	je	0x3c7508a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x68a>
 3c75074:      	movq	0x1b8(%rsp), %rdi
 3c7507c:      	movl	$0x1, %edx
 3c75081:      	movq	%r12, %rsi
 3c75084:      	callq	*0x14cbcb6(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c7508a:      	movq	0x218(%rsp), %r12
 3c75092:      	cmpq	$-0x2, %r12
 3c75096:      	je	0x3c750d1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c75098:      	cmpq	$-0x1, 0x230(%rsp)
 3c750a1:      	je	0x3c750b0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6b0>
 3c750a3:      	leaq	0x230(%rsp), %rdi
 3c750ab:      	callq	0x3c08af0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c750b0:      	cmpq	$-0x1, %r12
 3c750b4:      	je	0x3c750d1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c750b6:      	testq	%r12, %r12
 3c750b9:      	je	0x3c750d1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6d1>
 3c750bb:      	movq	0x220(%rsp), %rdi
 3c750c3:      	movl	$0x1, %edx
 3c750c8:      	movq	%r12, %rsi
 3c750cb:      	callq	*0x14cbc6f(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c750d1:      	cmpb	$0x0, 0x24(%rsp)
 3c750d6:      	jne	0x3c750f5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6f5>
 3c750d8:      	movq	0x14cbf89(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3c750df:      	movq	(%rax), %rax
 3c750e2:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c750ec:      	testq	%rcx, %rax
 3c750ef:      	jne	0x3c75195 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x795>
 3c750f5:      	xorl	%eax, %eax
 3c750f7:      	xchgl	%eax, (%r14)
 3c750fa:      	movb	$-0x1, %bl
 3c750fc:      	cmpl	$0x2, %eax
 3c750ff:      	jne	0x3c74f02 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c75105:      	movq	%r14, %rdi
 3c75108:      	callq	*0x14cbf62(%rip)        # 0x5141070 <writev+0x5141070>
 3c7510e:      	jmp	0x3c74f02 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x502>
 3c75113:      	movq	%r14, %rdi
 3c75116:      	callq	*0x14cc2ec(%rip)        # 0x5141408 <writev+0x5141408>
 3c7511c:      	jmp	0x3c74a30 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x30>
 3c75121:      	callq	*0x14cbf51(%rip)        # 0x5141078 <writev+0x5141078>
 3c75127:      	xorb	$0x1, %al
 3c75129:      	movzbl	0x4(%r14), %ecx
 3c7512e:      	testb	%cl, %cl
 3c75130:      	je	0x3c74a55 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x55>
 3c75136:      	movq	%r14, 0xc0(%rsp)
 3c7513e:      	movb	%al, 0xc8(%rsp)
 3c75145:      	leaq	-0x2e56723(%rip), %rdi  # 0xe1ea29 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c7514c:      	leaq	0x13b739d(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c75153:      	leaq	0x13b795e(%rip), %r8    # 0x502cab8 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x2f8>
 3c7515a:      	leaq	0xc0(%rsp), %rdx
 3c75162:      	movl	$0x2b, %esi
 3c75167:      	callq	*0x14cbd2b(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c7516d:      	ud2
 3c7516f:      	movq	%r14, %rdi
 3c75172:      	callq	*0x14cbef8(%rip)        # 0x5141070 <writev+0x5141070>
 3c75178:      	jmp	0x3c74ed4 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4d4>
 3c7517d:      	callq	*0x14cbef5(%rip)        # 0x5141078 <writev+0x5141078>
 3c75183:      	testb	%al, %al
 3c75185:      	jne	0x3c74ec6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c7518b:      	movb	$0x1, 0x4(%r14)
 3c75190:      	jmp	0x3c74ec6 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x4c6>
 3c75195:      	callq	*0x14cbedd(%rip)        # 0x5141078 <writev+0x5141078>
 3c7519b:      	testb	%al, %al
 3c7519d:      	jne	0x3c750f5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6f5>
 3c751a3:      	movb	$0x1, 0x4(%r14)
 3c751a8:      	jmp	0x3c750f5 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x6f5>
 3c751ad:      	movq	%rax, %r15
 3c751b0:      	jmp	0x3c75270 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x870>
 3c751b5:      	movq	%rax, %r15
 3c751b8:      	testq	%r12, %r12
 3c751bb:      	jle	0x3c7520b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x80b>
 3c751bd:      	movq	0x220(%rsp), %rdi
 3c751c5:      	movl	$0x1, %edx
 3c751ca:      	movq	%r12, %rsi
 3c751cd:      	callq	*0x14cbb6d(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c751d3:      	jmp	0x3c7520b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x80b>
 3c751d5:      	movq	%rax, %r15
 3c751d8:      	testq	%r12, %r12
 3c751db:      	jle	0x3c751f3 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x7f3>
 3c751dd:      	movq	0x1b8(%rsp), %rdi
 3c751e5:      	movl	$0x1, %edx
 3c751ea:      	movq	%r12, %rsi
 3c751ed:      	callq	*0x14cbb4d(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c751f3:      	cmpq	$-0x2, 0x218(%rsp)
 3c751fc:      	je	0x3c7520b <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x80b>
 3c751fe:      	leaq	0x218(%rsp), %rdi
 3c75206:      	callq	0x3c08980 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c7520b:      	movl	$0x0, 0x4(%rsp)
 3c75213:      	movq	%rbx, %r13
 3c75216:      	jmp	0x3c7525c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x85c>
 3c75218:      	callq	*0x14cbb9a(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c7521e:      	movq	%rax, %r15
 3c75221:      	movq	(%r13), %rsi
 3c75225:      	testq	%rsi, %rsi
 3c75228:      	jle	0x3c75278 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x878>
 3c7522a:      	movq	0x8(%r13), %rdi
 3c7522e:      	movl	$0x1, %edx
 3c75233:      	callq	*0x14cbb07(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c75239:      	movq	%r15, %rdi
 3c7523c:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c75241:      	movq	%rax, %r15
 3c75244:      	leaq	0xc0(%rsp), %rdi
 3c7524c:      	callq	0x3c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c75251:      	jmp	0x3c75270 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x870>
 3c75253:      	callq	*0x14cbb5f(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c75259:      	movq	%rax, %r15
 3c7525c:      	movzbl	0x24(%rsp), %esi
 3c75261:      	movq	%r14, %rdi
 3c75264:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c75269:      	cmpb	$0x0, 0x4(%rsp)
 3c7526e:      	je	0x3c75278 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore3put+0x878>
 3c75270:      	movq	%r13, %rdi
 3c75273:      	callq	0x3c08980 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server8metering9JobRecordEBF_>
 3c75278:      	movq	%r15, %rdi
 3c7527b:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c75280:      	callq	*0x14cbb32(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c75286:      	int3
 3c75287:      	int3
 3c75288:      	int3
 3c75289:      	int3
 3c7528a:      	int3
 3c7528b:      	int3
 3c7528c:      	int3
 3c7528d:      	int3
 3c7528e:      	int3
 3c7528f:      	int3

0000000003c75290 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take>:
 3c75290:      	pushq	%rbp
 3c75291:      	pushq	%r15
 3c75293:      	pushq	%r14
 3c75295:      	pushq	%r13
 3c75297:      	pushq	%r12
 3c75299:      	pushq	%rbx
 3c7529a:      	subq	$0x238, %rsp            # imm = 0x238
 3c752a1:      	movq	%rcx, %r15
 3c752a4:      	movq	%rdx, %r12
 3c752a7:      	movq	%rsi, %rbx
 3c752aa:      	movq	%rdi, %r13
 3c752ad:      	movl	$0x1, %ecx
 3c752b2:      	xorl	%eax, %eax
 3c752b4:      	lock
 3c752b5:      	cmpxchgl	%ecx, (%rsi)
 3c752b8:      	jne	0x3c755cf <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x33f>
 3c752be:      	movq	0x14cbda3(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c752c5:      	movq	(%r14), %rax
 3c752c8:      	shlq	%rax
 3c752cb:      	testq	%rax, %rax
 3c752ce:      	jne	0x3c755dd <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x34d>
 3c752d4:      	xorl	%eax, %eax
 3c752d6:      	movzbl	0x4(%rbx), %ecx
 3c752da:      	testb	%cl, %cl
 3c752dc:      	jne	0x3c755f1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x361>
 3c752e2:      	movl	%eax, 0x4(%rsp)
 3c752e6:      	leaq	0x8(%rbx), %rsi
 3c752ea:      	movq	%rbx, %rdi
 3c752ed:      	callq	0x3c47610 <_RNvMNtCs3pwlnhBXFtN_12memra_server9job_storeNtB2_16InMemoryJobStore12sweep_locked.llvm.12140772379724168186>
 3c752f2:      	cmpq	$0x0, 0x20(%rbx)
 3c752f7:      	je	0x3c7542a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x19a>
 3c752fd:      	leaq	0x28(%rbx), %rdi
 3c75301:      	movq	%r12, %rsi
 3c75304:      	movq	%r15, %rdx
 3c75307:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c7530c:      	movq	%rax, %rbp
 3c7530f:      	shrq	$0x39, %rax
 3c75313:      	movq	0x8(%rbx), %rcx
 3c75317:      	movq	0x10(%rbx), %rdx
 3c7531b:      	movd	%eax, %xmm0
 3c7531f:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 3c75323:      	pshuflw	$0x0, %xmm0, %xmm0      # xmm0 = xmm0[0,0,0,0,4,5,6,7]
 3c75328:      	pshufd	$0x44, %xmm0, %xmm1     # xmm1 = xmm0[0,1,0,1]
 3c7532d:      	xorl	%esi, %esi
 3c7532f:      	pcmpeqd	%xmm2, %xmm2
 3c75333:      	movq	0x14cba6e(%rip), %r8    # 0x5140da8 <writev+0x5140da8>
 3c7533a:      	andq	%rdx, %rbp
 3c7533d:      	movdqu	(%rcx,%rbp), %xmm3
 3c75342:      	movdqa	%xmm3, %xmm0
 3c75346:      	pcmpeqb	%xmm1, %xmm0
 3c7534a:      	pmovmskb	%xmm0, %eax
 3c7534e:      	testl	%eax, %eax
 3c75350:      	je	0x3c753f0 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x160>
 3c75356:      	movq	%r13, 0x8(%rsp)
 3c7535b:      	movq	%rcx, 0x20(%rsp)
 3c75360:      	movq	%rdx, 0x18(%rsp)
 3c75365:      	movdqa	%xmm1, 0x40(%rsp)
 3c7536b:      	movq	%rsi, 0x10(%rsp)
 3c75370:      	movdqa	%xmm3, 0x30(%rsp)
 3c75376:      	movq	%rax, 0x28(%rsp)
 3c7537b:      	tzcntl	%eax, %eax
 3c7537f:      	addq	%rbp, %rax
 3c75382:      	andq	%rdx, %rax
 3c75385:      	shlq	$0x8, %rax
 3c75389:      	movq	%rcx, %r13
 3c7538c:      	subq	%rax, %r13
 3c7538f:      	cmpq	-0xf0(%r13), %r15
 3c75396:      	jne	0x3c753be <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x12e>
 3c75398:      	movq	-0xf8(%r13), %rsi
 3c7539f:      	movq	%r12, %rdi
 3c753a2:      	movq	%r15, %rdx
 3c753a5:      	movq	%r15, %r14
 3c753a8:      	movq	%r12, %r15
 3c753ab:      	movq	%r8, %r12
 3c753ae:      	callq	*%r8
 3c753b1:      	movq	%r12, %r8
 3c753b4:      	movq	%r15, %r12
 3c753b7:      	movq	%r14, %r15
 3c753ba:      	testl	%eax, %eax
 3c753bc:      	je	0x3c75413 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x183>
 3c753be:      	movq	0x28(%rsp), %rcx
 3c753c3:      	leal	-0x1(%rcx), %eax
 3c753c6:      	andw	%cx, %ax
 3c753c9:      	movq	0x8(%rsp), %r13
 3c753ce:      	movq	0x20(%rsp), %rcx
 3c753d3:      	movq	0x18(%rsp), %rdx
 3c753d8:      	movdqa	0x40(%rsp), %xmm1
 3c753de:      	movq	0x10(%rsp), %rsi
 3c753e3:      	pcmpeqd	%xmm2, %xmm2
 3c753e7:      	movdqa	0x30(%rsp), %xmm3
 3c753ed:      	jne	0x3c75376 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0xe6>
 3c753ef:      	nop
 3c753f0:      	pcmpeqb	%xmm2, %xmm3
 3c753f4:      	pmovmskb	%xmm3, %eax
 3c753f8:      	testl	%eax, %eax
 3c753fa:      	movq	0x14cbc67(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c75401:      	jne	0x3c7542a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x19a>
 3c75403:      	addq	%rsi, %rbp
 3c75406:      	addq	$0x10, %rbp
 3c7540a:      	addq	$0x10, %rsi
 3c7540e:      	jmp	0x3c7533a <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0xaa>
 3c75413:      	cmpq	$-0x2, -0x80(%r13)
 3c75418:      	movq	0x8(%rsp), %r13
 3c7541d:      	movq	0x14cbc44(%rip), %r14   # 0x5141068 <writev+0x5141068>
 3c75424:      	jne	0x3c75590 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x300>
 3c7542a:      	leaq	0x28(%rbx), %rdi
 3c7542e:      	movq	%r12, %rsi
 3c75431:      	movq	%r15, %rdx
 3c75434:      	callq	0x3e68ed0 <_RINvYNtNtNtCs2AWtUsOyxgP_3std4hash6random11RandomStateNtNtCs4NRVxsYgnAr_4core4hash11BuildHasher8hash_oneReECs3pwlnhBXFtN_12memra_server>
 3c75439:      	leaq	0x58(%rsp), %rdi
 3c7543e:      	leaq	0x8(%rbx), %rsi
 3c75442:      	movq	%rax, %rdx
 3c75445:      	movq	%r12, %rcx
 3c75448:      	movq	%r15, %r8
 3c7544b:      	callq	0x389afb0 <_RINvMs6_NtCsgQfI1edjipl_9hashbrown3rawINtB6_8RawTableTNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs3pwlnhBXFtN_12memra_server9job_store5EntryEE12remove_entryNCINvNtB8_3map14equivalent_keyeBQ_B1r_E0EB1v_>
 3c75450:      	movq	0x58(%rsp), %r15
 3c75455:      	cmpq	$-0x1, %r15
 3c75459:      	je	0x3c7557d <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x2ed>
 3c7545f:      	movq	0x70(%rsp), %r12
 3c75464:      	leaq	0x78(%rsp), %rsi
 3c75469:      	leaq	0x158(%rsp), %rdi
 3c75471:      	movl	$0xe0, %edx
 3c75476:      	callq	*0x14cb8bc(%rip)        # 0x5140d38 <writev+0x5140d38>
 3c7547c:      	testq	%r15, %r15
 3c7547f:      	je	0x3c75494 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x204>
 3c75481:      	movq	0x60(%rsp), %rdi
 3c75486:      	movl	$0x1, %edx
 3c7548b:      	movq	%r15, %rsi
 3c7548e:      	callq	*0x14cb8ac(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c75494:      	cmpq	$-0x2, %r12
 3c75498:      	movl	0x4(%rsp), %ebp
 3c7549c:      	je	0x3c75581 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x2f1>
 3c754a2:      	movq	%r12, 0x58(%rsp)
 3c754a7:      	leaq	0x60(%rsp), %rdi
 3c754ac:      	leaq	0x158(%rsp), %rsi
 3c754b4:      	movl	$0xe0, %edx
 3c754b9:      	callq	*0x14cb879(%rip)        # 0x5140d38 <writev+0x5140d38>
 3c754bf:      	movq	0x38(%rbx), %rax
 3c754c3:      	xorl	%ecx, %ecx
 3c754c5:      	subq	0x138(%rsp), %rax
 3c754cd:      	cmovaeq	%rax, %rcx
 3c754d1:      	movq	%rcx, 0x38(%rbx)
 3c754d5:      	movq	0xc0(%rsp), %r15
 3c754dd:      	cmpq	$-0x2, %r15
 3c754e1:      	je	0x3c7551c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x28c>
 3c754e3:      	cmpq	$-0x1, 0xd8(%rsp)
 3c754ec:      	je	0x3c754fb <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x26b>
 3c754ee:      	leaq	0xd8(%rsp), %rdi
 3c754f6:      	callq	0x3c08af0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs8OSp0AlFmbY_10serde_json5value5ValueECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c754fb:      	cmpq	$-0x1, %r15
 3c754ff:      	je	0x3c7551c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x28c>
 3c75501:      	testq	%r15, %r15
 3c75504:      	je	0x3c7551c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x28c>
 3c75506:      	movq	0xc8(%rsp), %rdi
 3c7550e:      	movl	$0x1, %edx
 3c75513:      	movq	%r15, %rsi
 3c75516:      	callq	*0x14cb824(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c7551c:      	movq	%r12, (%r13)
 3c75520:      	movups	0x158(%rsp), %xmm0
 3c75528:      	movdqu	0x168(%rsp), %xmm1
 3c75531:      	movdqu	0x178(%rsp), %xmm2
 3c7553a:      	movdqu	0x188(%rsp), %xmm3
 3c75543:      	movups	%xmm0, 0x8(%r13)
 3c75548:      	movdqu	%xmm1, 0x18(%r13)
 3c7554e:      	movdqu	%xmm2, 0x28(%r13)
 3c75554:      	movdqu	%xmm3, 0x38(%r13)
 3c7555a:      	movups	0x198(%rsp), %xmm0
 3c75562:      	movups	%xmm0, 0x48(%r13)
 3c75567:      	movdqu	0x1a8(%rsp), %xmm0
 3c75570:      	movdqu	%xmm0, 0x58(%r13)
 3c75576:      	testb	%bpl, %bpl
 3c75579:      	je	0x3c7559f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x30f>
 3c7557b:      	jmp	0x3c755b1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c7557d:      	movl	0x4(%rsp), %ebp
 3c75581:      	movq	$-0x2, (%r13)
 3c75589:      	testb	%bpl, %bpl
 3c7558c:      	je	0x3c7559f <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x30f>
 3c7558e:      	jmp	0x3c755b1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c75590:      	movq	$-0x2, (%r13)
 3c75598:      	cmpb	$0x0, 0x4(%rsp)
 3c7559d:      	jne	0x3c755b1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c7559f:      	movq	(%r14), %rax
 3c755a2:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3c755ac:      	testq	%rcx, %rax
 3c755af:      	jne	0x3c7562c <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x39c>
 3c755b1:      	xorl	%eax, %eax
 3c755b3:      	xchgl	%eax, (%rbx)
 3c755b5:      	cmpl	$0x2, %eax
 3c755b8:      	je	0x3c75621 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x391>
 3c755ba:      	movq	%r13, %rax
 3c755bd:      	addq	$0x238, %rsp            # imm = 0x238
 3c755c4:      	popq	%rbx
 3c755c5:      	popq	%r12
 3c755c7:      	popq	%r13
 3c755c9:      	popq	%r14
 3c755cb:      	popq	%r15
 3c755cd:      	popq	%rbp
 3c755ce:      	retq
 3c755cf:      	movq	%rbx, %rdi
 3c755d2:      	callq	*0x14cbe30(%rip)        # 0x5141408 <writev+0x5141408>
 3c755d8:      	jmp	0x3c752be <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x2e>
 3c755dd:      	callq	*0x14cba95(%rip)        # 0x5141078 <writev+0x5141078>
 3c755e3:      	xorb	$0x1, %al
 3c755e5:      	movzbl	0x4(%rbx), %ecx
 3c755e9:      	testb	%cl, %cl
 3c755eb:      	je	0x3c752e2 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x52>
 3c755f1:      	movq	%rbx, 0x58(%rsp)
 3c755f6:      	movb	%al, 0x60(%rsp)
 3c755fa:      	leaq	-0x2e56bd8(%rip), %rdi  # 0xe1ea29 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.248.llvm.12140772379724168186>
 3c75601:      	leaq	0x13b6ee8(%rip), %rcx   # 0x502c4f0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.250.llvm.12140772379724168186>
 3c75608:      	leaq	0x13b74c1(%rip), %r8    # 0x502cad0 <anon.b731ebf5b89b152c0d2ad556fb42dc5b.295.llvm.12140772379724168186+0x310>
 3c7560f:      	leaq	0x58(%rsp), %rdx
 3c75614:      	movl	$0x2b, %esi
 3c75619:      	callq	*0x14cb879(%rip)        # 0x5140e98 <writev+0x5140e98>
 3c7561f:      	ud2
 3c75621:      	movq	%rbx, %rdi
 3c75624:      	callq	*0x14cba46(%rip)        # 0x5141070 <writev+0x5141070>
 3c7562a:      	jmp	0x3c755ba <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x32a>
 3c7562c:      	callq	*0x14cba46(%rip)        # 0x5141078 <writev+0x5141078>
 3c75632:      	testb	%al, %al
 3c75634:      	jne	0x3c755b1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c7563a:      	movb	$0x1, 0x4(%rbx)
 3c7563e:      	jmp	0x3c755b1 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x321>
 3c75643:      	movq	%rax, %r14
 3c75646:      	testq	%r15, %r15
 3c75649:      	jle	0x3c75666 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x3d6>
 3c7564b:      	movq	0xc8(%rsp), %rdi
 3c75653:      	movl	$0x1, %edx
 3c75658:      	movq	%r15, %rsi
 3c7565b:      	callq	*0x14cb6df(%rip)        # 0x5140d40 <writev+0x5140d40>
 3c75661:      	jmp	0x3c75666 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x3d6>
 3c75663:      	movq	%rax, %r14
 3c75666:      	movzbl	0x4(%rsp), %esi
 3c7566b:      	movq	%rbx, %rdi
 3c7566e:      	callq	0x3c04750 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardNtNtCs3pwlnhBXFtN_12memra_server9job_store5InnerEEB1A_.llvm.12140772379724168186>
 3c75673:      	jmp	0x3c75688 <_RNvXs_NtCs3pwlnhBXFtN_12memra_server9job_storeNtB4_16InMemoryJobStoreNtNtB6_8metering8JobStore4take+0x3f8>
 3c75675:      	callq	*0x14cb73d(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c7567b:      	movq	%rax, %r14
 3c7567e:      	leaq	0x58(%rsp), %rdi
 3c75683:      	callq	0x3c03370 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtBE_5mutex10MutexGuardjEEECs3pwlnhBXFtN_12memra_server.llvm.12140772379724168186>
 3c75688:      	movq	%r14, %rdi
 3c7568b:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3c75690:      	callq	*0x14cb722(%rip)        # 0x5140db8 <writev+0x5140db8>
 3c75696:      	int3
 3c75697:      	int3
 3c75698:      	int3
 3c75699:      	int3
 3c7569a:      	int3
 3c7569b:      	int3
 3c7569c:      	int3
 3c7569d:      	int3
 3c7569e:      	int3
 3c7569f:      	int3

0000000003ce3f70 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on>:
 3ce3f70:      	pushq	%rbp
 3ce3f71:      	pushq	%r15
 3ce3f73:      	pushq	%r14
 3ce3f75:      	pushq	%r13
 3ce3f77:      	pushq	%r12
 3ce3f79:      	pushq	%rbx
 3ce3f7a:      	subq	$0xe78, %rsp            # imm = 0xE78
 3ce3f81:      	movq	%r8, %r14
 3ce3f84:      	movl	%edx, 0x24(%rsp)
 3ce3f88:      	movq	%r9, 0xa8(%rsp)
 3ce3f90:      	movq	0x18(%rcx), %rbp
 3ce3f94:      	testq	%rbp, %rbp
 3ce3f97:      	movq	%rdi, 0x30(%rsp)
 3ce3f9c:      	je	0x3ce43a2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x432>
 3ce3fa2:      	movq	0x10(%r14), %rax
 3ce3fa6:      	movq	%rax, 0x70(%rsp)
 3ce3fab:      	movups	(%r14), %xmm0
 3ce3faf:      	movaps	%xmm0, 0x60(%rsp)
 3ce3fb4:      	movq	%r9, 0xb0(%rsp)
 3ce3fbc:      	movq	0x28(%rbp), %r14
 3ce3fc0:      	movl	0x146a11a(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x10>
 3ce3fc6:      	testl	%eax, %eax
 3ce3fc8:      	jne	0x3ce52df <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x136f>
 3ce3fce:      	leaq	0x10(%rbp), %rax
 3ce3fd2:      	movq	%rax, 0x40(%rsp)
 3ce3fd7:      	movzbl	0x146a0f2(%rip), %ecx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 3ce3fde:      	movq	0x146a0f3(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x8>
 3ce3fe5:      	movq	%r14, %rdx
 3ce3fe8:      	shrq	$0x3e, %rdx
 3ce3fec:      	movzbl	0x24(%rsp), %ebx
 3ce3ff1:      	jne	0x3ce52f0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1380>
 3ce3ff7:      	leaq	(,%r14,4), %rdx
 3ce3fff:      	testb	%cl, %cl
 3ce4001:      	cmovneq	%rax, %rdx
 3ce4005:      	movq	%rdx, 0x48(%rsp)
 3ce400a:      	testq	%r14, %r14
 3ce400d:      	je	0x3ce5366 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13f6>
 3ce4013:      	movq	%rbx, 0x38(%rsp)
 3ce4018:      	nopl	(%rax,%rax)
 3ce4020:      	movq	0x30(%rbp,%rbx,8), %rax
 3ce4025:      	movq	%rax, 0x50(%rsp)
 3ce402a:      	movq	0x30(%rbp), %rbx
 3ce402e:      	movq	0x38(%rbp), %r12
 3ce4032:      	movq	0x40(%rbp), %r15
 3ce4036:      	movl	0x146a0d4(%rip), %eax   # 0x514e110 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891+0x8>
 3ce403c:      	testl	%eax, %eax
 3ce403e:      	jne	0x3ce419a <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x22a>
 3ce4044:      	addq	%rbx, %r12
 3ce4047:      	addq	%r15, %r12
 3ce404a:      	movq	0x146a0b7(%rip), %rsi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891>
 3ce4051:      	leaq	0x10(%rbp), %rdi
 3ce4055:      	callq	0x39b5bc0 <_RNvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB4_9RouteLoad18service_estimate_s>
 3ce405a:      	movq	%rax, %rcx
 3ce405d:      	movq	%r12, %rax
 3ce4060:      	orq	%r14, %rax
 3ce4063:      	shrq	$0x20, %rax
 3ce4067:      	je	0x3ce4160 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1f0>
 3ce406d:      	movq	%r12, %rax
 3ce4070:      	xorl	%edx, %edx
 3ce4072:      	divq	%r14
 3ce4075:      	movq	%rax, %rdx
 3ce4078:      	incq	%rdx
 3ce407b:      	movq	%rcx, %rax
 3ce407e:      	mulq	%rdx
 3ce4081:      	jo	0x3ce4179 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x209>
 3ce4087:      	movq	%rax, %r13
 3ce408a:      	movq	%r13, 0x28(%rsp)
 3ce408f:      	movq	0x50(%rsp), %r15
 3ce4094:      	cmpq	0x48(%rsp), %r15
 3ce4099:      	jae	0x3ce481b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x8ab>
 3ce409f:      	movq	0x60(%rbp), %rbx
 3ce40a3:      	callq	*0x145e037(%rip)        # 0x51420e0 <writev+0x51420e0>
 3ce40a9:      	leaq	0x68(%rsp), %rdi
 3ce40ae:      	movq	%rax, %rsi
 3ce40b1:      	callq	*0x145e039(%rip)        # 0x51420f0 <writev+0x51420f0>
 3ce40b7:      	imulq	$0x3e8, %rax, %rax      # imm = 0x3E8
 3ce40be:      	movl	%edx, %ecx
 3ce40c0:      	imulq	$0x431bde83, %rcx, %rcx # imm = 0x431BDE83
 3ce40c7:      	shrq	$0x32, %rcx
 3ce40cb:      	addq	%rax, %rcx
 3ce40ce:      	movq	%rcx, 0x58(%rsp)
 3ce40d3:      	cmpb	$0x0, 0x24(%rsp)
 3ce40d8:      	jne	0x3ce4112 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1a2>
 3ce40da:      	addq	%r12, %rbx
 3ce40dd:      	cmpq	%r14, %rbx
 3ce40e0:      	jb	0x3ce4112 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1a2>
 3ce40e2:      	movq	%r13, %rax
 3ce40e5:      	movl	$0x3e8, %edx            # imm = 0x3E8
 3ce40ea:      	mulq	%rdx
 3ce40ed:      	jo	0x3ce41ab <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x23b>
 3ce40f3:      	cmpq	%rcx, %rax
 3ce40f6:      	ja	0x3ce41bb <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x24b>
 3ce40fc:      	movq	0xb0(%rsp), %rax
 3ce4104:      	testq	%rax, %rax
 3ce4107:      	je	0x3ce4112 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1a2>
 3ce4109:      	cmpq	%rax, %r13
 3ce410c:      	ja	0x3ce4d2c <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xdbc>
 3ce4112:      	leaq	0x1(%r15), %rcx
 3ce4116:      	movq	%r15, %rax
 3ce4119:      	movq	0x38(%rsp), %rbx
 3ce411e:      	lock
 3ce411f:      	cmpxchgq	%rcx, 0x30(%rbp,%rbx,8)
 3ce4125:      	jne	0x3ce4020 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xb0>
 3ce412b:      	lock
 3ce412c:      	incq	(%rbp)
 3ce4130:      	jle	0x3ce5364 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13f4>
 3ce4136:      	movq	%rbp, 0xc0(%rsp)
 3ce413e:      	callq	*0x145cd6c(%rip)        # 0x5140eb0 <writev+0x5140eb0>
 3ce4144:      	cmpl	$-0x1, %edx
 3ce4147:      	je	0x3ce4020 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xb0>
 3ce414d:      	jmp	0x3ce4cd7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xd67>
 3ce4152:      	nopw	%cs:(%rax,%rax)
 3ce4160:      	movl	%r12d, %eax
 3ce4163:      	xorl	%edx, %edx
 3ce4165:      	divl	%r14d
 3ce4168:      	movl	%eax, %edx
 3ce416a:      	incq	%rdx
 3ce416d:      	movq	%rcx, %rax
 3ce4170:      	mulq	%rdx
 3ce4173:      	jno	0x3ce4087 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x117>
 3ce4179:      	movq	$-0x1, %r13
 3ce4180:      	movq	%r13, 0x28(%rsp)
 3ce4185:      	movq	0x50(%rsp), %r15
 3ce418a:      	cmpq	0x48(%rsp), %r15
 3ce418f:      	jb	0x3ce409f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x12f>
 3ce4195:      	jmp	0x3ce481b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x8ab>
 3ce419a:      	leaq	0x1469f67(%rip), %rdi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891>
 3ce41a1:      	callq	0x3b2c932 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s0E0zEB1w_>
 3ce41a6:      	jmp	0x3ce4044 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xd4>
 3ce41ab:      	movq	$-0x1, %rax
 3ce41b2:      	cmpq	%rcx, %rax
 3ce41b5:      	jbe	0x3ce40fc <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x18c>
 3ce41bb:      	movups	0x18(%rbp), %xmm0
 3ce41bf:      	movups	%xmm0, 0x5f0(%rsp)
 3ce41c7:      	leaq	0x28(%rsp), %rax
 3ce41cc:      	movq	%rax, 0xc0(%rsp)
 3ce41d4:      	movq	0x145cce5(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce41db:      	movq	%rax, 0xc8(%rsp)
 3ce41e3:      	leaq	0x5f0(%rsp), %rcx
 3ce41eb:      	movq	%rcx, 0xd0(%rsp)
 3ce41f3:      	leaq	-0x2de70a(%rip), %rcx   # 0x3a05af0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtReNtB6_5Debug3fmtCs3pwlnhBXFtN_12memra_server>
 3ce41fa:      	movq	%rcx, 0xd8(%rsp)
 3ce4202:      	leaq	0x58(%rsp), %rcx
 3ce4207:      	movq	%rcx, 0xe0(%rsp)
 3ce420f:      	movq	%rax, 0xe8(%rsp)
 3ce4217:      	leaq	-0x2ec3408(%rip), %rsi  # 0xe20e16 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x123>
 3ce421e:      	leaq	0x78(%rsp), %rdi
 3ce4223:      	leaq	0xc0(%rsp), %rdx
 3ce422b:      	callq	*0x145cb27(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4231:      	movq	0x28(%rsp), %rbx
 3ce4236:      	movq	0x80(%rsp), %r14
 3ce423e:      	movq	0x88(%rsp), %rdx
 3ce4246:      	leaq	-0x2ec3357(%rip), %r15  # 0xe20ef6 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x203>
 3ce424d:      	movq	%r15, 0x8(%rsp)
 3ce4252:      	movq	$0xd, 0x10(%rsp)
 3ce425b:      	leaq	-0x395d422(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.4933824046006920891>
 3ce4262:      	leaq	0xc0(%rsp), %rdi
 3ce426a:      	movl	$0x10, %r8d
 3ce4270:      	movq	%r14, %rsi
 3ce4273:      	xorl	%r9d, %r9d
 3ce4276:      	callq	0x3cbb490 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce427b:      	movq	0x100(%rsp), %rax
 3ce4283:      	movq	%rax, 0xa70(%rsp)
 3ce428b:      	movups	0xc0(%rsp), %xmm0
 3ce4293:      	movups	0xd0(%rsp), %xmm1
 3ce429b:      	movups	0xe0(%rsp), %xmm2
 3ce42a3:      	movups	0xf0(%rsp), %xmm3
 3ce42ab:      	movaps	%xmm3, 0xa60(%rsp)
 3ce42b3:      	movaps	%xmm2, 0xa50(%rsp)
 3ce42bb:      	movaps	%xmm1, 0xa40(%rsp)
 3ce42c3:      	movaps	%xmm0, 0xa30(%rsp)
 3ce42cb:      	leaq	0xc0(%rsp), %rdi
 3ce42d3:      	leaq	0xa30(%rsp), %rsi
 3ce42db:      	callq	0x3eb00a0 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce42e0:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce42ea:      	leaq	0x5f0(%rsp), %rdi
 3ce42f2:      	leaq	0xc0(%rsp), %rsi
 3ce42fa:      	movl	$0x1, %edx
 3ce42ff:      	movq	%rbx, %rcx
 3ce4302:      	callq	0x3ce1c80 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce4307:      	movups	0x660(%rsp), %xmm0
 3ce430f:      	movq	0x30(%rsp), %r12
 3ce4314:      	movups	%xmm0, 0x70(%r12)
 3ce431a:      	movups	0x650(%rsp), %xmm0
 3ce4322:      	movups	%xmm0, 0x60(%r12)
 3ce4328:      	movups	0x640(%rsp), %xmm0
 3ce4330:      	movups	%xmm0, 0x50(%r12)
 3ce4336:      	movups	0x630(%rsp), %xmm0
 3ce433e:      	movups	%xmm0, 0x40(%r12)
 3ce4344:      	movups	0x5f0(%rsp), %xmm0
 3ce434c:      	movups	0x600(%rsp), %xmm1
 3ce4354:      	movups	0x610(%rsp), %xmm2
 3ce435c:      	movups	0x620(%rsp), %xmm3
 3ce4364:      	movups	%xmm3, 0x30(%r12)
 3ce436a:      	movups	%xmm2, 0x20(%r12)
 3ce4370:      	movups	%xmm1, 0x10(%r12)
 3ce4376:      	movups	%xmm0, (%r12)
 3ce437b:      	movq	%r15, 0x80(%r12)
 3ce4383:      	movq	$0xd, 0x88(%r12)
 3ce438f:      	movq	0x78(%rsp), %rsi
 3ce4394:      	testq	%rsi, %rsi
 3ce4397:      	jne	0x3ce4f11 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfa1>
 3ce439d:      	jmp	0x3ce5100 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce43a2:      	movl	0x1469de0(%rip), %eax   # 0x514e188 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891+0x18>
 3ce43a8:      	testl	%eax, %eax
 3ce43aa:      	jne	0x3ce5307 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1397>
 3ce43b0:      	movzbl	0x24(%rsp), %r12d
 3ce43b6:      	leaq	0x1469db3(%rip), %rax   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 3ce43bd:      	movq	(%rax,%r12,8), %rbx
 3ce43c1:      	cmpq	$0x1, %rbx
 3ce43c5:      	movq	%rbx, %r13
 3ce43c8:      	adcq	$0x0, %r13
 3ce43cc:      	movl	0x1469d0e(%rip), %eax   # 0x514e0e0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x10>
 3ce43d2:      	testl	%eax, %eax
 3ce43d4:      	jne	0x3ce5324 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13b4>
 3ce43da:      	movq	0x1469cef(%rip), %rdx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 3ce43e1:      	movq	0x1469cf0(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x8>
 3ce43e8:      	shrq	$0x3e, %rbx
 3ce43ec:      	jne	0x3ce5358 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x13e8>
 3ce43f2:      	leaq	(,%r13,4), %rdi
 3ce43fa:      	testq	%rdx, %rdx
 3ce43fd:      	cmovneq	%rax, %rdi
 3ce4401:      	movq	%rdi, 0x28(%rsp)
 3ce4406:      	movq	0x40(%rsi), %rax
 3ce440a:      	leaq	0x10(%rax), %r15
 3ce440e:      	addq	$0x8, %r14
 3ce4412:      	movq	0x8(%rcx), %rcx
 3ce4416:      	movq	%rcx, 0x40(%rsp)
 3ce441b:      	movq	%rax, 0x38(%rsp)
 3ce4420:      	addq	$0x18, %rax
 3ce4424:      	movq	%rax, 0xb8(%rsp)
 3ce442c:      	leaq	0x5f0(%rsp), %rbp
 3ce4434:      	xorl	%eax, %eax
 3ce4436:      	movl	$0x1, %ecx
 3ce443b:      	lock
 3ce443c:      	cmpxchgl	%ecx, (%r15)
 3ce4440:      	jne	0x3ce449b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x52b>
 3ce4442:      	nopw	%cs:(%rax,%rax)
 3ce4450:      	movq	0x145cc11(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3ce4457:      	movq	(%rax), %rax
 3ce445a:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3ce4464:      	testq	%rcx, %rax
 3ce4467:      	jne	0x3ce44a6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x536>
 3ce4469:      	xorl	%ebx, %ebx
 3ce446b:      	movq	0x38(%rsp), %rax
 3ce4470:      	movzbl	0x14(%rax), %eax
 3ce4474:      	testb	%al, %al
 3ce4476:      	je	0x3ce44c0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x550>
 3ce4478:      	movq	%r15, 0x5f8(%rsp)
 3ce4480:      	movb	%bl, 0x600(%rsp)
 3ce4487:      	movq	$0x0, 0x5f0(%rsp)
 3ce4493:      	movq	%rbp, %rbx
 3ce4496:      	jmp	0x3ce451d <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x5ad>
 3ce449b:      	movq	%r15, %rdi
 3ce449e:      	callq	*0x145cf64(%rip)        # 0x5141408 <writev+0x5141408>
 3ce44a4:      	jmp	0x3ce4450 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x4e0>
 3ce44a6:      	callq	*0x145cbcc(%rip)        # 0x5141078 <writev+0x5141078>
 3ce44ac:      	movl	%eax, %ebx
 3ce44ae:      	xorb	$0x1, %bl
 3ce44b1:      	movq	0x38(%rsp), %rax
 3ce44b6:      	movzbl	0x14(%rax), %eax
 3ce44ba:      	testb	%al, %al
 3ce44bc:      	jne	0x3ce4478 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x508>
 3ce44be:      	nop
 3ce44c0:      	leaq	0xc0(%rsp), %rdi
 3ce44c8:      	movq	0xb8(%rsp), %rsi
 3ce44d0:      	callq	0x3cf9b00 <_RNvXs1M_NtCs3pwlnhBXFtN_12memra_server6workerNtB6_7MetricsNtNtCs4NRVxsYgnAr_4core5clone5Clone5clone>
 3ce44d5:      	testb	%bl, %bl
 3ce44d7:      	jne	0x3ce44f6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x586>
 3ce44d9:      	movq	0x145cb88(%rip), %rax   # 0x5141068 <writev+0x5141068>
 3ce44e0:      	movq	(%rax), %rax
 3ce44e3:      	movabsq	$0x7fffffffffffffff, %rcx # imm = 0x7FFFFFFFFFFFFFFF
 3ce44ed:      	testq	%rcx, %rax
 3ce44f0:      	jne	0x3ce47ff <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x88f>
 3ce44f6:      	xorl	%eax, %eax
 3ce44f8:      	xchgl	%eax, (%r15)
 3ce44fb:      	cmpl	$0x2, %eax
 3ce44fe:      	movq	%rbp, %rbx
 3ce4501:      	je	0x3ce47f1 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x881>
 3ce4507:      	movl	$0x440, %edx            # imm = 0x440
 3ce450c:      	movq	%rbx, %rdi
 3ce450f:      	leaq	0xc0(%rsp), %rsi
 3ce4517:      	callq	*0x145c81b(%rip)        # 0x5140d38 <writev+0x5140d38>
 3ce451d:      	leaq	0xa30(%rsp), %rdi
 3ce4525:      	movq	%rbx, %rsi
 3ce4528:      	callq	0x3cf3bb0 <_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsINtNtNtCs2AWtUsOyxgP_3std4sync6poison11PoisonErrorINtNtB1v_5mutex10MutexGuardBH_EEE17unwrap_or_defaultBL_>
 3ce452d:      	movq	0xeb0(%rsp), %rax
 3ce4535:      	movq	(%rax,%r12,8), %rbx
 3ce4539:      	movq	%rbx, 0x58(%rsp)
 3ce453e:      	movq	0xb88(%rsp), %rax
 3ce4546:      	testq	%rax, %rax
 3ce4549:      	je	0x3ce4610 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6a0>
 3ce454f:      	movss	0xe68(%rsp), %xmm0
 3ce4558:      	xorps	%xmm1, %xmm1
 3ce455b:      	ucomiss	%xmm1, %xmm0
 3ce455e:      	jbe	0x3ce4610 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6a0>
 3ce4564:      	movsd	0xb90(%rsp), %xmm1
 3ce456d:      	movsd	-0x3963c75(%rip), %xmm3 # 0x380900 <anon.78da417e2a41163eb606632dc77243d1.14.llvm.14221926250047164484+0x60>
 3ce4575:      	unpcklps	%xmm3, %xmm1            # xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
 3ce4578:      	movapd	-0x3965580(%rip), %xmm4 # 0x37f000 <writev+0x37f000>
 3ce4580:      	subpd	%xmm4, %xmm1
 3ce4584:      	movapd	%xmm1, %xmm2
 3ce4588:      	unpckhpd	%xmm1, %xmm2            # xmm2 = xmm2[1],xmm1[1]
 3ce458c:      	addsd	%xmm1, %xmm2
 3ce4590:      	movq	%rax, %xmm1
 3ce4595:      	punpckldq	%xmm3, %xmm1    # xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
 3ce4599:      	subpd	%xmm4, %xmm1
 3ce459d:      	movapd	%xmm1, %xmm3
 3ce45a1:      	unpckhpd	%xmm1, %xmm3            # xmm3 = xmm3[1],xmm1[1]
 3ce45a5:      	addsd	%xmm1, %xmm3
 3ce45a9:      	divsd	%xmm3, %xmm2
 3ce45ad:      	cvtss2sd	%xmm0, %xmm0
 3ce45b1:      	mulsd	%xmm2, %xmm0
 3ce45b5:      	divsd	-0x395839d(%rip), %xmm0 # 0x38c220 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.580.llvm.6417388326552730234+0x20>
 3ce45bd:      	callq	*0x145cd35(%rip)        # 0x51412f8 <writev+0x51412f8>
 3ce45c3:      	movapd	%xmm0, %xmm1
 3ce45c7:      	subsd	-0x395a147(%rip), %xmm1 # 0x38a488 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.1436.llvm.6417388326552730234+0x40>
 3ce45cf:      	xorpd	%xmm2, %xmm2
 3ce45d3:      	ucomisd	%xmm2, %xmm0
 3ce45d7:      	jb	0x3ce4770 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x800>
 3ce45dd:      	cvttsd2si	%xmm0, %rax
 3ce45e2:      	movq	%rax, %rdx
 3ce45e5:      	sarq	$0x3f, %rdx
 3ce45e9:      	cvttsd2si	%xmm1, %rcx
 3ce45ee:      	andq	%rdx, %rcx
 3ce45f1:      	orq	%rax, %rcx
 3ce45f4:      	ucomisd	-0x395baac(%rip), %xmm0 # 0x388b50 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.1382.llvm.6417388326552730234+0x28>
 3ce45fc:      	movq	$-0x1, %rax
 3ce4603:      	jbe	0x3ce4783 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x813>
 3ce4609:      	jmp	0x3ce4786 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x816>
 3ce460e:      	nop
 3ce4610:      	movl	0x1469afa(%rip), %eax   # 0x514e110 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891+0x8>
 3ce4616:      	testl	%eax, %eax
 3ce4618:      	jne	0x3ce47cb <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x85b>
 3ce461e:      	movq	0x1469ae3(%rip), %rcx   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891>
 3ce4625:      	movq	%rbx, %rax
 3ce4628:      	orq	%r13, %rax
 3ce462b:      	shrq	$0x20, %rax
 3ce462f:      	je	0x3ce4730 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x7c0>
 3ce4635:      	movq	%rbx, %rax
 3ce4638:      	xorl	%edx, %edx
 3ce463a:      	divq	%r13
 3ce463d:      	movq	%rax, %rdx
 3ce4640:      	incq	%rdx
 3ce4643:      	movq	%rcx, %rax
 3ce4646:      	mulq	%rdx
 3ce4649:      	jo	0x3ce4748 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x7d8>
 3ce464f:      	movq	%rax, 0x60(%rsp)
 3ce4654:      	cmpq	0x28(%rsp), %rbx
 3ce4659:      	jae	0x3ce4a6b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xafb>
 3ce465f:      	callq	*0x145da7b(%rip)        # 0x51420e0 <writev+0x51420e0>
 3ce4665:      	movq	%r14, %rdi
 3ce4668:      	movq	%rax, %rsi
 3ce466b:      	callq	*0x145da7f(%rip)        # 0x51420f0 <writev+0x51420f0>
 3ce4671:      	cmpq	$0x0, 0x40(%rsp)
 3ce4677:      	sete	%cl
 3ce467a:      	imulq	$0x3e8, %rax, %rax      # imm = 0x3E8
 3ce4681:      	movl	%edx, %edx
 3ce4683:      	imulq	$0x431bde83, %rdx, %rsi # imm = 0x431BDE83
 3ce468a:      	shrq	$0x32, %rsi
 3ce468e:      	addq	%rax, %rsi
 3ce4691:      	testq	%rbx, %rbx
 3ce4694:      	setne	%al
 3ce4697:      	movq	%rsi, 0x78(%rsp)
 3ce469c:      	cmpb	$0x0, 0x24(%rsp)
 3ce46a1:      	jne	0x3ce46e2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x772>
 3ce46a3:      	orb	%al, %cl
 3ce46a5:      	je	0x3ce46e2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x772>
 3ce46a7:      	movq	0x60(%rsp), %rcx
 3ce46ac:      	movq	%rcx, %rax
 3ce46af:      	movl	$0x3e8, %edx            # imm = 0x3E8
 3ce46b4:      	mulq	%rdx
 3ce46b7:      	jo	0x3ce47dc <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x86c>
 3ce46bd:      	cmpq	%rsi, %rax
 3ce46c0:      	ja	0x3ce4f24 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfb4>
 3ce46c6:      	movq	0xa8(%rsp), %rax
 3ce46ce:      	testq	%rax, %rax
 3ce46d1:      	setne	%dl
 3ce46d4:      	cmpq	%rax, %rcx
 3ce46d7:      	seta	%al
 3ce46da:      	testb	%al, %dl
 3ce46dc:      	jne	0x3ce5115 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x11a5>
 3ce46e2:      	leaq	0x1(%rbx), %rcx
 3ce46e6:      	movq	%rbx, %rax
 3ce46e9:      	movq	0xeb0(%rsp), %rdx
 3ce46f1:      	lock
 3ce46f2:      	cmpxchgq	%rcx, (%rdx,%r12,8)
 3ce46f7:      	je	0x3ce4c8d <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xd1d>
 3ce46fd:      	leaq	0xa30(%rsp), %rdi
 3ce4705:      	callq	0x3c90c00 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEBF_>
 3ce470a:      	xorl	%eax, %eax
 3ce470c:      	movl	$0x1, %ecx
 3ce4711:      	lock
 3ce4712:      	cmpxchgl	%ecx, (%r15)
 3ce4716:      	je	0x3ce4450 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x4e0>
 3ce471c:      	jmp	0x3ce449b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x52b>
 3ce4721:      	nopw	%cs:(%rax,%rax)
 3ce4730:      	movl	%ebx, %eax
 3ce4732:      	xorl	%edx, %edx
 3ce4734:      	divl	%r13d
 3ce4737:      	movl	%eax, %edx
 3ce4739:      	incq	%rdx
 3ce473c:      	movq	%rcx, %rax
 3ce473f:      	mulq	%rdx
 3ce4742:      	jno	0x3ce464f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6df>
 3ce4748:      	movq	$-0x1, %rax
 3ce474f:      	movq	%rax, 0x60(%rsp)
 3ce4754:      	cmpq	0x28(%rsp), %rbx
 3ce4759:      	jb	0x3ce465f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6ef>
 3ce475f:      	jmp	0x3ce4a6b <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xafb>
 3ce4764:      	nopw	%cs:(%rax,%rax)
 3ce4770:      	xorl	%ecx, %ecx
 3ce4772:      	ucomisd	-0x395bc2a(%rip), %xmm0 # 0x388b50 <anon.1c761382ec4b0d6a0d4e6a3322c5a8e8.1382.llvm.6417388326552730234+0x28>
 3ce477a:      	movq	$-0x1, %rax
 3ce4781:      	ja	0x3ce4786 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x816>
 3ce4783:      	movq	%rcx, %rax
 3ce4786:      	movq	%rax, %rdx
 3ce4789:      	cmpq	$0x258, %rax            # imm = 0x258
 3ce478f:      	jae	0x3ce47b0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x840>
 3ce4791:      	movl	$0x1, %ecx
 3ce4796:      	testq	%rax, %rax
 3ce4799:      	je	0x3ce4625 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6b5>
 3ce479f:      	jmp	0x3ce47c3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x853>
 3ce47a1:      	nopw	%cs:(%rax,%rax)
 3ce47b0:      	movl	$0x258, %edx            # imm = 0x258
 3ce47b5:      	movl	$0x1, %ecx
 3ce47ba:      	testq	%rax, %rax
 3ce47bd:      	je	0x3ce4625 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6b5>
 3ce47c3:      	movq	%rdx, %rcx
 3ce47c6:      	jmp	0x3ce4625 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6b5>
 3ce47cb:      	leaq	0x1469936(%rip), %rdi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891>
 3ce47d2:      	callq	0x3b2c932 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s0E0zEB1w_>
 3ce47d7:      	jmp	0x3ce461e <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x6ae>
 3ce47dc:      	movq	$-0x1, %rax
 3ce47e3:      	cmpq	%rsi, %rax
 3ce47e6:      	jbe	0x3ce46c6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x756>
 3ce47ec:      	jmp	0x3ce4f24 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfb4>
 3ce47f1:      	movq	%r15, %rdi
 3ce47f4:      	callq	*0x145c876(%rip)        # 0x5141070 <writev+0x5141070>
 3ce47fa:      	jmp	0x3ce4507 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x597>
 3ce47ff:      	callq	*0x145c873(%rip)        # 0x5141078 <writev+0x5141078>
 3ce4805:      	testb	%al, %al
 3ce4807:      	jne	0x3ce44f6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x586>
 3ce480d:      	movq	0x38(%rsp), %rax
 3ce4812:      	movb	$0x1, 0x14(%rax)
 3ce4816:      	jmp	0x3ce44f6 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x586>
 3ce481b:      	movq	0x38(%rsp), %rdx
 3ce4820:      	shll	$0x3, %edx
 3ce4823:      	leaq	0x13772ee(%rip), %rax   # 0x505bb18 <anon.d42d8d1710be02f83480d8869995a290.3835.llvm.4933824046006920891+0xf0>
 3ce482a:      	movq	(%rdx,%rax), %rax
 3ce482e:      	leaq	-0x2eacaad(%rip), %rcx  # 0xe37d88 <anon.d42d8d1710be02f83480d8869995a290.3778.llvm.4933824046006920891+0x219>
 3ce4835:      	movq	(%rdx,%rcx), %rcx
 3ce4839:      	movq	%rax, 0xa30(%rsp)
 3ce4841:      	movq	%rcx, 0xa38(%rsp)
 3ce4849:      	movups	0x18(%rbp), %xmm0
 3ce484d:      	movups	%xmm0, 0x5f0(%rsp)
 3ce4855:      	leaq	0xa30(%rsp), %rax
 3ce485d:      	movq	%rax, 0xc0(%rsp)
 3ce4865:      	leaq	-0x2debdc(%rip), %rax   # 0x3a05c90 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtCs3pwlnhBXFtN_12memra_server>
 3ce486c:      	movq	%rax, 0xc8(%rsp)
 3ce4874:      	leaq	0x5f0(%rsp), %rax
 3ce487c:      	movq	%rax, 0xd0(%rsp)
 3ce4884:      	leaq	-0x2ded9b(%rip), %rax   # 0x3a05af0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtReNtB6_5Debug3fmtCs3pwlnhBXFtN_12memra_server>
 3ce488b:      	movq	%rax, 0xd8(%rsp)
 3ce4893:      	leaq	0x50(%rsp), %rax
 3ce4898:      	movq	%rax, 0xe0(%rsp)
 3ce48a0:      	movq	0x145c4c1(%rip), %rax   # 0x5140d68 <writev+0x5140d68>
 3ce48a7:      	movq	%rax, 0xe8(%rsp)
 3ce48af:      	leaq	0x48(%rsp), %rcx
 3ce48b4:      	movq	%rcx, 0xf0(%rsp)
 3ce48bc:      	movq	%rax, 0xf8(%rsp)
 3ce48c4:      	leaq	0x28(%rsp), %rax
 3ce48c9:      	movq	%rax, 0x100(%rsp)
 3ce48d1:      	movq	0x145c5e8(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce48d8:      	movq	%rax, 0x108(%rsp)
 3ce48e0:      	leaq	-0x390277c(%rip), %rsi  # 0x3e216b <anon.fa2e791037615afd04c732fea021b56d.22.llvm.7881675451276153+0xc4>
 3ce48e7:      	leaq	0x78(%rsp), %rdi
 3ce48ec:      	leaq	0xc0(%rsp), %rdx
 3ce48f4:      	callq	*0x145c45e(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce48fa:      	movq	0x28(%rsp), %rbx
 3ce48ff:      	movq	0x80(%rsp), %r14
 3ce4907:      	movq	0x88(%rsp), %rdx
 3ce490f:      	leaq	-0x2ec3a13(%rip), %r15  # 0xe20f03 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x210>
 3ce4916:      	movq	%r15, 0x8(%rsp)
 3ce491b:      	movq	$0xa, 0x10(%rsp)
 3ce4924:      	leaq	-0x395daeb(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.4933824046006920891>
 3ce492b:      	leaq	0xc0(%rsp), %rdi
 3ce4933:      	movl	$0x10, %r8d
 3ce4939:      	movq	%r14, %rsi
 3ce493c:      	xorl	%r9d, %r9d
 3ce493f:      	callq	0x3cbb490 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4944:      	movq	0x100(%rsp), %rax
 3ce494c:      	movq	%rax, 0xa70(%rsp)
 3ce4954:      	movups	0xc0(%rsp), %xmm0
 3ce495c:      	movups	0xd0(%rsp), %xmm1
 3ce4964:      	movups	0xe0(%rsp), %xmm2
 3ce496c:      	movups	0xf0(%rsp), %xmm3
 3ce4974:      	movaps	%xmm3, 0xa60(%rsp)
 3ce497c:      	movaps	%xmm2, 0xa50(%rsp)
 3ce4984:      	movaps	%xmm1, 0xa40(%rsp)
 3ce498c:      	movaps	%xmm0, 0xa30(%rsp)
 3ce4994:      	leaq	0xc0(%rsp), %rdi
 3ce499c:      	leaq	0xa30(%rsp), %rsi
 3ce49a4:      	callq	0x3eb00a0 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce49a9:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce49b3:      	leaq	0x5f0(%rsp), %rdi
 3ce49bb:      	leaq	0xc0(%rsp), %rsi
 3ce49c3:      	movl	$0x1, %edx
 3ce49c8:      	movq	%rbx, %rcx
 3ce49cb:      	callq	0x3ce1c80 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce49d0:      	movups	0x660(%rsp), %xmm0
 3ce49d8:      	movq	0x30(%rsp), %r12
 3ce49dd:      	movups	%xmm0, 0x70(%r12)
 3ce49e3:      	movups	0x650(%rsp), %xmm0
 3ce49eb:      	movups	%xmm0, 0x60(%r12)
 3ce49f1:      	movups	0x640(%rsp), %xmm0
 3ce49f9:      	movups	%xmm0, 0x50(%r12)
 3ce49ff:      	movups	0x630(%rsp), %xmm0
 3ce4a07:      	movups	%xmm0, 0x40(%r12)
 3ce4a0d:      	movups	0x5f0(%rsp), %xmm0
 3ce4a15:      	movups	0x600(%rsp), %xmm1
 3ce4a1d:      	movups	0x610(%rsp), %xmm2
 3ce4a25:      	movups	0x620(%rsp), %xmm3
 3ce4a2d:      	movups	%xmm3, 0x30(%r12)
 3ce4a33:      	movups	%xmm2, 0x20(%r12)
 3ce4a39:      	movups	%xmm1, 0x10(%r12)
 3ce4a3f:      	movups	%xmm0, (%r12)
 3ce4a44:      	movq	%r15, 0x80(%r12)
 3ce4a4c:      	movq	$0xa, 0x88(%r12)
 3ce4a58:      	movq	0x78(%rsp), %rsi
 3ce4a5d:      	testq	%rsi, %rsi
 3ce4a60:      	jne	0x3ce4f11 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xfa1>
 3ce4a66:      	jmp	0x3ce5100 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4a6b:      	shll	$0x3, %r12d
 3ce4a6f:      	leaq	0x13770a2(%rip), %rax   # 0x505bb18 <anon.d42d8d1710be02f83480d8869995a290.3835.llvm.4933824046006920891+0xf0>
 3ce4a76:      	movq	(%r12,%rax), %rax
 3ce4a7a:      	leaq	-0x2eaccf9(%rip), %rcx  # 0xe37d88 <anon.d42d8d1710be02f83480d8869995a290.3778.llvm.4933824046006920891+0x219>
 3ce4a81:      	movq	(%r12,%rcx), %rcx
 3ce4a85:      	movq	%rax, 0x78(%rsp)
 3ce4a8a:      	movq	%rcx, 0x80(%rsp)
 3ce4a92:      	leaq	0x78(%rsp), %rax
 3ce4a97:      	movq	%rax, 0xc0(%rsp)
 3ce4a9f:      	leaq	-0x2dee16(%rip), %rax   # 0x3a05c90 <_RNvXs1i_NtCs4NRVxsYgnAr_4core3fmtReNtB6_7Display3fmtCs3pwlnhBXFtN_12memra_server>
 3ce4aa6:      	movq	%rax, 0xc8(%rsp)
 3ce4aae:      	leaq	0x58(%rsp), %rax
 3ce4ab3:      	movq	%rax, 0xd0(%rsp)
 3ce4abb:      	movq	0x145c2a6(%rip), %rax   # 0x5140d68 <writev+0x5140d68>
 3ce4ac2:      	movq	%rax, 0xd8(%rsp)
 3ce4aca:      	leaq	0x28(%rsp), %rcx
 3ce4acf:      	movq	%rcx, 0xe0(%rsp)
 3ce4ad7:      	movq	%rax, 0xe8(%rsp)
 3ce4adf:      	leaq	0x60(%rsp), %rax
 3ce4ae4:      	movq	%rax, 0xf0(%rsp)
 3ce4aec:      	movq	0x145c3cd(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce4af3:      	movq	%rax, 0xf8(%rsp)
 3ce4afb:      	leaq	-0x391c5fe(%rip), %rsi  # 0x3c8504 <anon.8d7079b8273dd0721474669710734b31.74.llvm.16532073193574447039+0x101>
 3ce4b02:      	leaq	0x5f0(%rsp), %rdi
 3ce4b0a:      	leaq	0xc0(%rsp), %rdx
 3ce4b12:      	callq	*0x145c240(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4b18:      	movq	0x5f0(%rsp), %r14
 3ce4b20:      	movq	0x5f8(%rsp), %r15
 3ce4b28:      	movq	0x600(%rsp), %rdx
 3ce4b30:      	leaq	-0x2ec3c34(%rip), %rbx  # 0xe20f03 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x210>
 3ce4b37:      	movq	%rbx, 0x8(%rsp)
 3ce4b3c:      	movq	$0xa, 0x10(%rsp)
 3ce4b45:      	leaq	-0x395dd0c(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.4933824046006920891>
 3ce4b4c:      	leaq	0xc0(%rsp), %rdi
 3ce4b54:      	movl	$0x10, %r8d
 3ce4b5a:      	movq	%r15, %rsi
 3ce4b5d:      	xorl	%r9d, %r9d
 3ce4b60:      	callq	0x3cbb490 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4b65:      	movq	0x30(%rsp), %r12
 3ce4b6a:      	movq	0x100(%rsp), %rax
 3ce4b72:      	movq	%rax, 0x540(%rsp)
 3ce4b7a:      	movups	0xc0(%rsp), %xmm0
 3ce4b82:      	movups	0xd0(%rsp), %xmm1
 3ce4b8a:      	movupd	0xe0(%rsp), %xmm2
 3ce4b93:      	movupd	0xf0(%rsp), %xmm3
 3ce4b9c:      	movapd	%xmm3, 0x530(%rsp)
 3ce4ba5:      	movapd	%xmm2, 0x520(%rsp)
 3ce4bae:      	movaps	%xmm1, 0x510(%rsp)
 3ce4bb6:      	movaps	%xmm0, 0x500(%rsp)
 3ce4bbe:      	leaq	0xc0(%rsp), %rdi
 3ce4bc6:      	leaq	0x500(%rsp), %rsi
 3ce4bce:      	callq	0x3eb00a0 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce4bd3:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce4bdd:      	movq	0x60(%rsp), %rcx
 3ce4be2:      	leaq	0x5f0(%rsp), %rdi
 3ce4bea:      	leaq	0xc0(%rsp), %rsi
 3ce4bf2:      	movl	$0x1, %edx
 3ce4bf7:      	callq	0x3ce1c80 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce4bfc:      	movups	0x660(%rsp), %xmm0
 3ce4c04:      	movups	%xmm0, 0x70(%r12)
 3ce4c0a:      	movups	0x650(%rsp), %xmm0
 3ce4c12:      	movups	%xmm0, 0x60(%r12)
 3ce4c18:      	movups	0x640(%rsp), %xmm0
 3ce4c20:      	movups	%xmm0, 0x50(%r12)
 3ce4c26:      	movups	0x630(%rsp), %xmm0
 3ce4c2e:      	movups	%xmm0, 0x40(%r12)
 3ce4c34:      	movups	0x5f0(%rsp), %xmm0
 3ce4c3c:      	movups	0x600(%rsp), %xmm1
 3ce4c44:      	movups	0x610(%rsp), %xmm2
 3ce4c4c:      	movups	0x620(%rsp), %xmm3
 3ce4c54:      	movups	%xmm3, 0x30(%r12)
 3ce4c5a:      	movups	%xmm2, 0x20(%r12)
 3ce4c60:      	movups	%xmm1, 0x10(%r12)
 3ce4c66:      	movups	%xmm0, (%r12)
 3ce4c6b:      	movq	%rbx, 0x80(%r12)
 3ce4c73:      	movq	$0xa, 0x88(%r12)
 3ce4c7f:      	testq	%r14, %r14
 3ce4c82:      	jne	0x3ce50e2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1172>
 3ce4c88:      	jmp	0x3ce50f3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce4c8d:      	movq	0xeb8(%rsp), %rax
 3ce4c95:      	lock
 3ce4c96:      	incq	(%rax)
 3ce4c99:      	movq	0x30(%rsp), %r12
 3ce4c9e:      	movq	0xeb0(%rsp), %rcx
 3ce4ca6:      	movq	%rcx, 0x8(%r12)
 3ce4cab:      	movq	%rax, 0x10(%r12)
 3ce4cb0:      	movl	$0xffffffff, 0x30(%r12) # imm = 0xFFFFFFFF
 3ce4cb9:      	movl	0x24(%rsp), %eax
 3ce4cbd:      	movb	%al, 0x38(%r12)
 3ce4cc2:      	movw	$0x1, 0x39(%r12)
 3ce4cca:      	movq	$-0x1, (%r12)
 3ce4cd2:      	jmp	0x3ce50f3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce4cd7:      	movq	0xeb8(%rsp), %rcx
 3ce4cdf:      	lock
 3ce4ce0:      	incq	(%rcx)
 3ce4ce3:      	movq	0x30(%rsp), %r12
 3ce4ce8:      	movq	0xeb0(%rsp), %rsi
 3ce4cf0:      	movq	%rsi, 0x8(%r12)
 3ce4cf5:      	movq	%rcx, 0x10(%r12)
 3ce4cfa:      	movq	%rbx, 0x18(%r12)
 3ce4cff:      	movq	%rbp, 0x20(%r12)
 3ce4d04:      	movq	%rax, 0x28(%r12)
 3ce4d09:      	movl	%edx, 0x30(%r12)
 3ce4d0e:      	movl	0x24(%rsp), %eax
 3ce4d12:      	movb	%al, 0x38(%r12)
 3ce4d17:      	movw	$0x101, 0x39(%r12)      # imm = 0x101
 3ce4d1f:      	movq	$-0x1, (%r12)
 3ce4d27:      	jmp	0x3ce5100 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4d2c:      	movups	0x18(%rbp), %xmm0
 3ce4d30:      	movups	%xmm0, 0x5f0(%rsp)
 3ce4d38:      	leaq	0x28(%rsp), %rax
 3ce4d3d:      	movq	%rax, 0xc0(%rsp)
 3ce4d45:      	movq	0x145c174(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce4d4c:      	movq	%rax, 0xc8(%rsp)
 3ce4d54:      	leaq	0x5f0(%rsp), %rcx
 3ce4d5c:      	movq	%rcx, 0xd0(%rsp)
 3ce4d64:      	leaq	-0x2df27b(%rip), %rcx   # 0x3a05af0 <_RNvXs1g_NtCs4NRVxsYgnAr_4core3fmtReNtB6_5Debug3fmtCs3pwlnhBXFtN_12memra_server>
 3ce4d6b:      	movq	%rcx, 0xd8(%rsp)
 3ce4d73:      	leaq	0xb0(%rsp), %rcx
 3ce4d7b:      	movq	%rcx, 0xe0(%rsp)
 3ce4d83:      	movq	%rax, 0xe8(%rsp)
 3ce4d8b:      	leaq	-0x2ec404d(%rip), %rsi  # 0xe20d45 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x52>
 3ce4d92:      	leaq	0x78(%rsp), %rdi
 3ce4d97:      	leaq	0xc0(%rsp), %rdx
 3ce4d9f:      	callq	*0x145bfb3(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4da5:      	movq	0x28(%rsp), %rbx
 3ce4daa:      	movq	0x80(%rsp), %r14
 3ce4db2:      	movq	0x88(%rsp), %rdx
 3ce4dba:      	leaq	-0x2ec3fba(%rip), %r15  # 0xe20e07 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x114>
 3ce4dc1:      	movq	%r15, 0x8(%rsp)
 3ce4dc6:      	movq	$0xf, 0x10(%rsp)
 3ce4dcf:      	leaq	-0x395df96(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.4933824046006920891>
 3ce4dd6:      	leaq	0xc0(%rsp), %rdi
 3ce4dde:      	movl	$0x10, %r8d
 3ce4de4:      	movq	%r14, %rsi
 3ce4de7:      	xorl	%r9d, %r9d
 3ce4dea:      	callq	0x3cbb490 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4def:      	movq	0x100(%rsp), %rax
 3ce4df7:      	movq	%rax, 0xa70(%rsp)
 3ce4dff:      	movups	0xc0(%rsp), %xmm0
 3ce4e07:      	movups	0xd0(%rsp), %xmm1
 3ce4e0f:      	movups	0xe0(%rsp), %xmm2
 3ce4e17:      	movups	0xf0(%rsp), %xmm3
 3ce4e1f:      	movaps	%xmm3, 0xa60(%rsp)
 3ce4e27:      	movaps	%xmm2, 0xa50(%rsp)
 3ce4e2f:      	movaps	%xmm1, 0xa40(%rsp)
 3ce4e37:      	movaps	%xmm0, 0xa30(%rsp)
 3ce4e3f:      	leaq	0xc0(%rsp), %rdi
 3ce4e47:      	leaq	0xa30(%rsp), %rsi
 3ce4e4f:      	callq	0x3eb00a0 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce4e54:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce4e5e:      	leaq	0x5f0(%rsp), %rdi
 3ce4e66:      	leaq	0xc0(%rsp), %rsi
 3ce4e6e:      	movl	$0x1, %edx
 3ce4e73:      	movq	%rbx, %rcx
 3ce4e76:      	callq	0x3ce1c80 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce4e7b:      	movups	0x660(%rsp), %xmm0
 3ce4e83:      	movq	0x30(%rsp), %r12
 3ce4e88:      	movups	%xmm0, 0x70(%r12)
 3ce4e8e:      	movups	0x650(%rsp), %xmm0
 3ce4e96:      	movups	%xmm0, 0x60(%r12)
 3ce4e9c:      	movups	0x640(%rsp), %xmm0
 3ce4ea4:      	movups	%xmm0, 0x50(%r12)
 3ce4eaa:      	movups	0x630(%rsp), %xmm0
 3ce4eb2:      	movups	%xmm0, 0x40(%r12)
 3ce4eb8:      	movups	0x5f0(%rsp), %xmm0
 3ce4ec0:      	movups	0x600(%rsp), %xmm1
 3ce4ec8:      	movups	0x610(%rsp), %xmm2
 3ce4ed0:      	movups	0x620(%rsp), %xmm3
 3ce4ed8:      	movups	%xmm3, 0x30(%r12)
 3ce4ede:      	movups	%xmm2, 0x20(%r12)
 3ce4ee4:      	movups	%xmm1, 0x10(%r12)
 3ce4eea:      	movups	%xmm0, (%r12)
 3ce4eef:      	movq	%r15, 0x80(%r12)
 3ce4ef7:      	movq	$0xf, 0x88(%r12)
 3ce4f03:      	movq	0x78(%rsp), %rsi
 3ce4f08:      	testq	%rsi, %rsi
 3ce4f0b:      	je	0x3ce5100 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4f11:      	movl	$0x1, %edx
 3ce4f16:      	movq	%r14, %rdi
 3ce4f19:      	callq	*0x145be21(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce4f1f:      	jmp	0x3ce5100 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1190>
 3ce4f24:      	leaq	0x60(%rsp), %rax
 3ce4f29:      	movq	%rax, 0xc0(%rsp)
 3ce4f31:      	movq	0x145bf88(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce4f38:      	movq	%rax, 0xc8(%rsp)
 3ce4f40:      	leaq	0x78(%rsp), %rcx
 3ce4f45:      	movq	%rcx, 0xd0(%rsp)
 3ce4f4d:      	movq	%rax, 0xd8(%rsp)
 3ce4f55:      	leaq	-0x2ec3266(%rip), %rsi  # 0xe21cf6 <anon.d42d8d1710be02f83480d8869995a290.1045.llvm.4933824046006920891+0x89>
 3ce4f5c:      	leaq	0x5f0(%rsp), %rdi
 3ce4f64:      	leaq	0xc0(%rsp), %rdx
 3ce4f6c:      	callq	*0x145bde6(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce4f72:      	movq	0x5f0(%rsp), %r14
 3ce4f7a:      	movq	0x5f8(%rsp), %r15
 3ce4f82:      	movq	0x600(%rsp), %rdx
 3ce4f8a:      	leaq	-0x2ec409b(%rip), %rbx  # 0xe20ef6 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x203>
 3ce4f91:      	movq	%rbx, 0x8(%rsp)
 3ce4f96:      	movq	$0xd, 0x10(%rsp)
 3ce4f9f:      	leaq	-0x395e166(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.4933824046006920891>
 3ce4fa6:      	leaq	0xc0(%rsp), %rdi
 3ce4fae:      	movl	$0x10, %r8d
 3ce4fb4:      	movq	%r15, %rsi
 3ce4fb7:      	xorl	%r9d, %r9d
 3ce4fba:      	callq	0x3cbb490 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce4fbf:      	movq	0x30(%rsp), %r12
 3ce4fc4:      	movq	0x100(%rsp), %rax
 3ce4fcc:      	movq	%rax, 0x590(%rsp)
 3ce4fd4:      	movups	0xc0(%rsp), %xmm0
 3ce4fdc:      	movups	0xd0(%rsp), %xmm1
 3ce4fe4:      	movupd	0xe0(%rsp), %xmm2
 3ce4fed:      	movupd	0xf0(%rsp), %xmm3
 3ce4ff6:      	movapd	%xmm3, 0x580(%rsp)
 3ce4fff:      	movapd	%xmm2, 0x570(%rsp)
 3ce5008:      	movaps	%xmm1, 0x560(%rsp)
 3ce5010:      	movaps	%xmm0, 0x550(%rsp)
 3ce5018:      	leaq	0xc0(%rsp), %rdi
 3ce5020:      	leaq	0x550(%rsp), %rsi
 3ce5028:      	callq	0x3eb00a0 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce502d:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce5037:      	movq	0x60(%rsp), %rcx
 3ce503c:      	leaq	0x5f0(%rsp), %rdi
 3ce5044:      	leaq	0xc0(%rsp), %rsi
 3ce504c:      	movl	$0x1, %edx
 3ce5051:      	callq	0x3ce1c80 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce5056:      	movups	0x660(%rsp), %xmm0
 3ce505e:      	movups	%xmm0, 0x70(%r12)
 3ce5064:      	movups	0x650(%rsp), %xmm0
 3ce506c:      	movups	%xmm0, 0x60(%r12)
 3ce5072:      	movups	0x640(%rsp), %xmm0
 3ce507a:      	movups	%xmm0, 0x50(%r12)
 3ce5080:      	movups	0x630(%rsp), %xmm0
 3ce5088:      	movups	%xmm0, 0x40(%r12)
 3ce508e:      	movups	0x5f0(%rsp), %xmm0
 3ce5096:      	movups	0x600(%rsp), %xmm1
 3ce509e:      	movupd	0x610(%rsp), %xmm2
 3ce50a7:      	movupd	0x620(%rsp), %xmm3
 3ce50b0:      	movupd	%xmm3, 0x30(%r12)
 3ce50b7:      	movupd	%xmm2, 0x20(%r12)
 3ce50be:      	movups	%xmm1, 0x10(%r12)
 3ce50c4:      	movups	%xmm0, (%r12)
 3ce50c9:      	movq	%rbx, 0x80(%r12)
 3ce50d1:      	movq	$0xd, 0x88(%r12)
 3ce50dd:      	testq	%r14, %r14
 3ce50e0:      	je	0x3ce50f3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce50e2:      	movl	$0x1, %edx
 3ce50e7:      	movq	%r15, %rdi
 3ce50ea:      	movq	%r14, %rsi
 3ce50ed:      	callq	*0x145bc4d(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce50f3:      	leaq	0xa30(%rsp), %rdi
 3ce50fb:      	callq	0x3c90c00 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEBF_>
 3ce5100:      	movq	%r12, %rax
 3ce5103:      	addq	$0xe78, %rsp            # imm = 0xE78
 3ce510a:      	popq	%rbx
 3ce510b:      	popq	%r12
 3ce510d:      	popq	%r13
 3ce510f:      	popq	%r14
 3ce5111:      	popq	%r15
 3ce5113:      	popq	%rbp
 3ce5114:      	retq
 3ce5115:      	leaq	0x60(%rsp), %rax
 3ce511a:      	movq	%rax, 0xc0(%rsp)
 3ce5122:      	movq	0x145bd97(%rip), %rax   # 0x5140ec0 <writev+0x5140ec0>
 3ce5129:      	movq	%rax, 0xc8(%rsp)
 3ce5131:      	leaq	0xa8(%rsp), %rcx
 3ce5139:      	movq	%rcx, 0xd0(%rsp)
 3ce5141:      	movq	%rax, 0xd8(%rsp)
 3ce5149:      	leaq	-0x2ec3386(%rip), %rsi  # 0xe21dca <anon.d42d8d1710be02f83480d8869995a290.1045.llvm.4933824046006920891+0x15d>
 3ce5150:      	leaq	0x5f0(%rsp), %rdi
 3ce5158:      	leaq	0xc0(%rsp), %rdx
 3ce5160:      	callq	*0x145bbf2(%rip)        # 0x5140d58 <writev+0x5140d58>
 3ce5166:      	movq	0x5f0(%rsp), %r14
 3ce516e:      	movq	0x5f8(%rsp), %r15
 3ce5176:      	movq	0x600(%rsp), %rdx
 3ce517e:      	leaq	-0x2ec437e(%rip), %rbx  # 0xe20e07 <anon.d42d8d1710be02f83480d8869995a290.867.llvm.4933824046006920891+0x114>
 3ce5185:      	movq	%rbx, 0x8(%rsp)
 3ce518a:      	movq	$0xf, 0x10(%rsp)
 3ce5193:      	leaq	-0x395e35a(%rip), %rcx  # 0x386e40 <anon.d42d8d1710be02f83480d8869995a290.585.llvm.4933824046006920891>
 3ce519a:      	leaq	0xc0(%rsp), %rdi
 3ce51a2:      	movl	$0x10, %r8d
 3ce51a8:      	movq	%r15, %rsi
 3ce51ab:      	xorl	%r9d, %r9d
 3ce51ae:      	callq	0x3cbb490 <_RNvCs3pwlnhBXFtN_12memra_server10error_body>
 3ce51b3:      	movq	0x30(%rsp), %r12
 3ce51b8:      	movq	0x100(%rsp), %rax
 3ce51c0:      	movq	%rax, 0x5e0(%rsp)
 3ce51c8:      	movups	0xc0(%rsp), %xmm0
 3ce51d0:      	movups	0xd0(%rsp), %xmm1
 3ce51d8:      	movupd	0xe0(%rsp), %xmm2
 3ce51e1:      	movupd	0xf0(%rsp), %xmm3
 3ce51ea:      	movapd	%xmm3, 0x5d0(%rsp)
 3ce51f3:      	movapd	%xmm2, 0x5c0(%rsp)
 3ce51fc:      	movaps	%xmm1, 0x5b0(%rsp)
 3ce5204:      	movaps	%xmm0, 0x5a0(%rsp)
 3ce520c:      	leaq	0xc0(%rsp), %rdi
 3ce5214:      	leaq	0x5a0(%rsp), %rsi
 3ce521c:      	callq	0x3eb00a0 <_RNvXs0_NtCs4Uug9gnAtr7_4axum4jsonINtB5_4JsonNtNtCs8OSp0AlFmbY_10serde_json5value5ValueENtNtNtCsaSG9NyffgI5_9axum_core8response13into_response12IntoResponse13into_responseCs3pwlnhBXFtN_12memra_server>
 3ce5221:      	movw	$0x1ad, 0x128(%rsp)     # imm = 0x1AD
 3ce522b:      	movq	0x60(%rsp), %rcx
 3ce5230:      	leaq	0x5f0(%rsp), %rdi
 3ce5238:      	leaq	0xc0(%rsp), %rsi
 3ce5240:      	movl	$0x1, %edx
 3ce5245:      	callq	0x3ce1c80 <_RNvCs3pwlnhBXFtN_12memra_server23retry_contract_response>
 3ce524a:      	movups	0x660(%rsp), %xmm0
 3ce5252:      	movups	%xmm0, 0x70(%r12)
 3ce5258:      	movups	0x650(%rsp), %xmm0
 3ce5260:      	movups	%xmm0, 0x60(%r12)
 3ce5266:      	movups	0x640(%rsp), %xmm0
 3ce526e:      	movups	%xmm0, 0x50(%r12)
 3ce5274:      	movups	0x630(%rsp), %xmm0
 3ce527c:      	movups	%xmm0, 0x40(%r12)
 3ce5282:      	movups	0x5f0(%rsp), %xmm0
 3ce528a:      	movups	0x600(%rsp), %xmm1
 3ce5292:      	movupd	0x610(%rsp), %xmm2
 3ce529b:      	movupd	0x620(%rsp), %xmm3
 3ce52a4:      	movupd	%xmm3, 0x30(%r12)
 3ce52ab:      	movupd	%xmm2, 0x20(%r12)
 3ce52b2:      	movups	%xmm1, 0x10(%r12)
 3ce52b8:      	movups	%xmm0, (%r12)
 3ce52bd:      	movq	%rbx, 0x80(%r12)
 3ce52c5:      	movq	$0xf, 0x88(%r12)
 3ce52d1:      	testq	%r14, %r14
 3ce52d4:      	jne	0x3ce50e2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1172>
 3ce52da:      	jmp	0x3ce50f3 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1183>
 3ce52df:      	leaq	0x1468dea(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 3ce52e6:      	callq	0x3b2b3b1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 3ce52eb:      	jmp	0x3ce3fce <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x5e>
 3ce52f0:      	testb	%cl, %cl
 3ce52f2:      	movq	$-0x1, %rcx
 3ce52f9:      	cmovneq	%rax, %rcx
 3ce52fd:      	movq	%rcx, 0x48(%rsp)
 3ce5302:      	jmp	0x3ce4013 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0xa3>
 3ce5307:      	leaq	0x1468e62(%rip), %rdi   # 0x514e170 <_RNvNvCs3pwlnhBXFtN_12memra_server8lane_cap4CAPS.llvm.4933824046006920891>
 3ce530e:      	movq	%rcx, %rbx
 3ce5311:      	movq	%rsi, %r15
 3ce5314:      	callq	0x3b2b258 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockAjj3_E10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server8lane_cap0E0zEB1A_>
 3ce5319:      	movq	%r15, %rsi
 3ce531c:      	movq	%rbx, %rcx
 3ce531f:      	jmp	0x3ce43b0 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x440>
 3ce5324:      	leaq	0x1468da5(%rip), %rdi   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 3ce532b:      	movq	%rcx, 0x38(%rsp)
 3ce5330:      	movq	%rsi, %r15
 3ce5333:      	callq	0x3b2b3b1 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockINtNtCs4NRVxsYgnAr_4core6option6OptionjEE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server15max_queue_depth0E0zEB29_>
 3ce5338:      	movq	%r15, %rsi
 3ce533b:      	movq	0x38(%rsp), %rcx
 3ce5340:      	movq	0x1468d89(%rip), %rdx   # 0x514e0d0 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891>
 3ce5347:      	movq	0x1468d8a(%rip), %rax   # 0x514e0d8 <_RNvNvCs3pwlnhBXFtN_12memra_server15max_queue_depth1D.llvm.4933824046006920891+0x8>
 3ce534e:      	shrq	$0x3e, %rbx
 3ce5352:      	je	0x3ce43f2 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x482>
 3ce5358:      	movq	$-0x1, %rdi
 3ce535f:      	jmp	0x3ce43fa <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x48a>
 3ce5364:      	ud2
 3ce5366:      	movq	0x30(%rbp,%rbx,8), %rax
 3ce536b:      	movq	%rax, 0x50(%rsp)
 3ce5370:      	movq	0x30(%rbp), %rax
 3ce5374:      	movq	0x38(%rbp), %rax
 3ce5378:      	movq	0x40(%rbp), %rax
 3ce537c:      	movl	0x1468d8e(%rip), %eax   # 0x514e110 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891+0x8>
 3ce5382:      	testl	%eax, %eax
 3ce5384:      	jne	0x3ce53a4 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1434>
 3ce5386:      	movq	0x1468d7b(%rip), %rsi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891>
 3ce538d:      	movq	0x40(%rsp), %rdi
 3ce5392:      	callq	0x39b5bc0 <_RNvMs_NtCs3pwlnhBXFtN_12memra_server15route_telemetryNtB4_9RouteLoad18service_estimate_s>
 3ce5397:      	leaq	0x13489da(%rip), %rdi   # 0x502dd78 <anon.d42d8d1710be02f83480d8869995a290.849.llvm.4933824046006920891+0x58>
 3ce539e:      	callq	*0x145b9d4(%rip)        # 0x5140d78 <writev+0x5140d78>
 3ce53a4:      	leaq	0x1468d5d(%rip), %rdi   # 0x514e108 <_RNvNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s1D.llvm.4933824046006920891>
 3ce53ab:      	callq	0x3b2c932 <_RINvMNtNtCs2AWtUsOyxgP_3std4sync9once_lockINtB3_8OnceLockyE10initializeNCINvB2_11get_or_initNCNvCs3pwlnhBXFtN_12memra_server19rl_reset_fallback_s0E0zEB1w_>
 3ce53b0:      	jmp	0x3ce5386 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1416>
 3ce53b2:      	jmp	0x3ce53bc <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x144c>
 3ce53b4:      	jmp	0x3ce53bc <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x144c>
 3ce53b6:      	jmp	0x3ce542f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14bf>
 3ce53b8:      	jmp	0x3ce53d7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1467>
 3ce53ba:      	jmp	0x3ce53d7 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x1467>
 3ce53bc:      	movq	%rax, %r12
 3ce53bf:      	testq	%r14, %r14
 3ce53c2:      	je	0x3ce5432 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14c2>
 3ce53c4:      	movl	$0x1, %edx
 3ce53c9:      	movq	%r15, %rdi
 3ce53cc:      	movq	%r14, %rsi
 3ce53cf:      	callq	*0x145b96b(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce53d5:      	jmp	0x3ce5432 <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14c2>
 3ce53d7:      	movq	%rax, %r12
 3ce53da:      	movq	0x78(%rsp), %rsi
 3ce53df:      	testq	%rsi, %rsi
 3ce53e2:      	je	0x3ce543f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce53e4:      	movl	$0x1, %edx
 3ce53e9:      	movq	%r14, %rdi
 3ce53ec:      	callq	*0x145b94e(%rip)        # 0x5140d40 <writev+0x5140d40>
 3ce53f2:      	movq	%r12, %rdi
 3ce53f5:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3ce53fa:      	movq	%rax, %r12
 3ce53fd:      	movzbl	%bl, %esi
 3ce5400:      	movq	%r15, %rdi
 3ce5403:      	callq	0x3c89c80 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueINtNtNtNtCs2AWtUsOyxgP_3std4sync6poison5mutex10MutexGuardINtNtB4_6option6OptionNtNtNtCs8e8YMT7Sgk_5tokio4sync9semaphore20OwnedSemaphorePermitEEECs3pwlnhBXFtN_12memra_server>
 3ce5408:      	jmp	0x3ce543f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce540a:      	callq	*0x145b9a8(%rip)        # 0x5140db8 <writev+0x5140db8>
 3ce5410:      	movq	%rax, %r12
 3ce5413:      	lock
 3ce5414:      	decq	(%rbp)
 3ce5418:      	jne	0x3ce543f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce541a:      	leaq	0xc0(%rsp), %rdi
 3ce5422:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3ce5427:      	jmp	0x3ce543f <_RNvCs3pwlnhBXFtN_12memra_server24reserve_pending_admit_on+0x14cf>
 3ce5429:      	callq	*0x145b989(%rip)        # 0x5140db8 <writev+0x5140db8>
 3ce542f:      	movq	%rax, %r12
 3ce5432:      	leaq	0xa30(%rsp), %rdi
 3ce543a:      	callq	0x3c90c00 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtNtCs3pwlnhBXFtN_12memra_server6worker7MetricsEBF_>
 3ce543f:      	movq	%r12, %rdi
 3ce5442:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3ce5447:      	callq	*0x145b96b(%rip)        # 0x5140db8 <writev+0x5140db8>
 3ce544d:      	int3
 3ce544e:      	int3
 3ce544f:      	int3

0000000003d869d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 3d869d0:      	pushq	%r14
 3d869d2:      	pushq	%rbx
 3d869d3:      	pushq	%rax
 3d869d4:      	movq	%rdi, %rbx
 3d869d7:      	cmpb	$0x1, 0x31(%rdi)
 3d869db:      	jne	0x3d86a31 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3d869dd:      	movq	0x8(%rbx), %rcx
 3d869e1:      	movq	(%rcx), %rax
 3d869e4:      	nopw	%cs:(%rax,%rax)
 3d869f0:      	testq	%rax, %rax
 3d869f3:      	je	0x3d86a00 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 3d869f5:      	leaq	-0x1(%rax), %rdx
 3d869f9:      	lock
 3d869fa:      	cmpxchgq	%rdx, (%rcx)
 3d869fe:      	jne	0x3d869f0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 3d86a00:      	cmpb	$0x0, 0x32(%rbx)
 3d86a04:      	jne	0x3d86a31 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3d86a06:      	movq	(%rbx), %rcx
 3d86a09:      	movzbl	0x30(%rbx), %edx
 3d86a0d:      	movq	(%rcx,%rdx,8), %rax
 3d86a11:      	nopw	%cs:(%rax,%rax)
 3d86a20:      	testq	%rax, %rax
 3d86a23:      	je	0x3d86a31 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3d86a25:      	leaq	-0x1(%rax), %rsi
 3d86a29:      	lock
 3d86a2a:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 3d86a2f:      	jne	0x3d86a20 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 3d86a31:      	cmpl	$-0x1, 0x28(%rbx)
 3d86a35:      	je	0x3d86a7a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3d86a37:      	movq	0x10(%rbx), %rdi
 3d86a3b:      	cmpq	$0x2, %rdi
 3d86a3f:      	ja	0x3d86a82 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 3d86a41:      	movq	0x18(%rbx), %rcx
 3d86a45:      	addq	$0x18, %rbx
 3d86a49:      	movq	0x30(%rcx,%rdi,8), %rax
 3d86a4e:      	nop
 3d86a50:      	testq	%rax, %rax
 3d86a53:      	je	0x3d86a62 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 3d86a55:      	leaq	-0x1(%rax), %rdx
 3d86a59:      	lock
 3d86a5a:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 3d86a60:      	jne	0x3d86a50 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 3d86a62:      	movq	(%rbx), %rax
 3d86a65:      	lock
 3d86a66:      	decq	(%rax)
 3d86a69:      	jne	0x3d86a7a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3d86a6b:      	movq	%rbx, %rdi
 3d86a6e:      	addq	$0x8, %rsp
 3d86a72:      	popq	%rbx
 3d86a73:      	popq	%r14
 3d86a75:      	jmp	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3d86a7a:      	addq	$0x8, %rsp
 3d86a7e:      	popq	%rbx
 3d86a7f:      	popq	%r14
 3d86a81:      	retq
 3d86a82:      	leaq	0x129b447(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 3d86a89:      	movl	$0x3, %esi
 3d86a8e:      	callq	*0x13ba3e4(%rip)        # 0x5140e78 <writev+0x5140e78>
 3d86a94:      	ud2
 3d86a96:      	movq	%rax, %r14
 3d86a99:      	movq	0x18(%rbx), %rax
 3d86a9d:      	lock
 3d86a9e:      	decq	(%rax)
 3d86aa1:      	jne	0x3d86aaf <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 3d86aa3:      	addq	$0x18, %rbx
 3d86aa7:      	movq	%rbx, %rdi
 3d86aaa:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3d86aaf:      	movq	%r14, %rdi
 3d86ab2:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3d86ab7:      	callq	*0x13ba2fb(%rip)        # 0x5140db8 <writev+0x5140db8>
 3d86abd:      	int3
 3d86abe:      	int3
 3d86abf:      	int3

0000000003e589b0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_>:
 3e589b0:      	pushq	%r14
 3e589b2:      	pushq	%rbx
 3e589b3:      	pushq	%rax
 3e589b4:      	movq	%rdi, %rbx
 3e589b7:      	cmpb	$0x1, 0x31(%rdi)
 3e589bb:      	jne	0x3e58a11 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3e589bd:      	movq	0x8(%rbx), %rcx
 3e589c1:      	movq	(%rcx), %rax
 3e589c4:      	nopw	%cs:(%rax,%rax)
 3e589d0:      	testq	%rax, %rax
 3e589d3:      	je	0x3e589e0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x30>
 3e589d5:      	leaq	-0x1(%rax), %rdx
 3e589d9:      	lock
 3e589da:      	cmpxchgq	%rdx, (%rcx)
 3e589de:      	jne	0x3e589d0 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x20>
 3e589e0:      	cmpb	$0x0, 0x32(%rbx)
 3e589e4:      	jne	0x3e58a11 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3e589e6:      	movq	(%rbx), %rcx
 3e589e9:      	movzbl	0x30(%rbx), %edx
 3e589ed:      	movq	(%rcx,%rdx,8), %rax
 3e589f1:      	nopw	%cs:(%rax,%rax)
 3e58a00:      	testq	%rax, %rax
 3e58a03:      	je	0x3e58a11 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x61>
 3e58a05:      	leaq	-0x1(%rax), %rsi
 3e58a09:      	lock
 3e58a0a:      	cmpxchgq	%rsi, (%rcx,%rdx,8)
 3e58a0f:      	jne	0x3e58a00 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x50>
 3e58a11:      	cmpl	$-0x1, 0x28(%rbx)
 3e58a15:      	je	0x3e58a5a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3e58a17:      	movq	0x10(%rbx), %rdi
 3e58a1b:      	cmpq	$0x2, %rdi
 3e58a1f:      	ja	0x3e58a62 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xb2>
 3e58a21:      	movq	0x18(%rbx), %rcx
 3e58a25:      	addq	$0x18, %rbx
 3e58a29:      	movq	0x30(%rcx,%rdi,8), %rax
 3e58a2e:      	nop
 3e58a30:      	testq	%rax, %rax
 3e58a33:      	je	0x3e58a42 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x92>
 3e58a35:      	leaq	-0x1(%rax), %rdx
 3e58a39:      	lock
 3e58a3a:      	cmpxchgq	%rdx, 0x30(%rcx,%rdi,8)
 3e58a40:      	jne	0x3e58a30 <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0x80>
 3e58a42:      	movq	(%rbx), %rax
 3e58a45:      	lock
 3e58a46:      	decq	(%rax)
 3e58a49:      	jne	0x3e58a5a <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xaa>
 3e58a4b:      	movq	%rbx, %rdi
 3e58a4e:      	addq	$0x8, %rsp
 3e58a52:      	popq	%rbx
 3e58a53:      	popq	%r14
 3e58a55:      	jmp	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3e58a5a:      	addq	$0x8, %rsp
 3e58a5e:      	popq	%rbx
 3e58a5f:      	popq	%r14
 3e58a61:      	retq
 3e58a62:      	leaq	0x11c9467(%rip), %rdx   # 0x5021ed0 <anon.7def173938c2086f7773e60aae9b9908.1648.llvm.11839224041111125489>
 3e58a69:      	movl	$0x3, %esi
 3e58a6e:      	callq	*0x12e8404(%rip)        # 0x5140e78 <writev+0x5140e78>
 3e58a74:      	ud2
 3e58a76:      	movq	%rax, %r14
 3e58a79:      	movq	0x18(%rbx), %rax
 3e58a7d:      	lock
 3e58a7e:      	decq	(%rax)
 3e58a81:      	jne	0x3e58a8f <_RINvNtCs4NRVxsYgnAr_4core3ptr9drop_glueNtCs3pwlnhBXFtN_12memra_server21PendingAdmissionGuardEBD_+0xdf>
 3e58a83:      	addq	$0x18, %rbx
 3e58a87:      	movq	%rbx, %rdi
 3e58a8a:      	callq	0x39bcc70 <_RNvMsn_NtCscdodAO9FK5_5alloc4syncINtB5_3ArcNtNtCs3pwlnhBXFtN_12memra_server15route_telemetry9RouteLoadE9drop_slowBJ_>
 3e58a8f:      	movq	%r14, %rdi
 3e58a92:      	callq	0x4fde6d0 <_Unwind_Resume@plt>
 3e58a97:      	callq	*0x12e831b(%rip)        # 0x5140db8 <writev+0x5140db8>
 3e58a9d:      	int3
 3e58a9e:      	int3
 3e58a9f:      	int3

0000000003ebd050 <_RNvCs9wFQrvczXsK_7___rustc12___rust_alloc>:
 3ebd050:      	jmp	0x4f8f3f0 <_RNvCs9wFQrvczXsK_7___rustc11___rdl_alloc>
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

0000000003ebd060 <_RNvCs9wFQrvczXsK_7___rustc14___rust_dealloc>:
 3ebd060:      	jmp	0x4f8f450 <_RNvCs9wFQrvczXsK_7___rustc13___rdl_dealloc>
 3ebd065:      	int3
 3ebd066:      	int3
 3ebd067:      	int3
 3ebd068:      	int3
 3ebd069:      	int3
 3ebd06a:      	int3
 3ebd06b:      	int3
 3ebd06c:      	int3
 3ebd06d:      	int3
 3ebd06e:      	int3
 3ebd06f:      	int3

0000000003ebd090 <_RNvCs9wFQrvczXsK_7___rustc35___rust_no_alloc_shim_is_unstable_v2>:
 3ebd090:      	retq
 3ebd091:      	int3
 3ebd092:      	int3
 3ebd093:      	int3
 3ebd094:      	int3
 3ebd095:      	int3
 3ebd096:      	int3
 3ebd097:      	int3
 3ebd098:      	int3
 3ebd099:      	int3
 3ebd09a:      	int3
 3ebd09b:      	int3
 3ebd09c:      	int3
 3ebd09d:      	int3
 3ebd09e:      	int3
 3ebd09f:      	int3

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
 4c7ee52:      	leaq	-0x48fa499(%rip), %rsi  # 0x3849c0 <anon.d42d8d1710be02f83480d8869995a290.115.llvm.4933824046006920891+0x300>
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
 4fcf09e:      	movdqa	-0x4c47916(%rip), %xmm2 # 0x387790 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x470>
 4fcf0a6:      	pcmpgtb	%xmm2, %xmm0
 4fcf0aa:      	punpcklbw	%xmm0, %xmm0    # xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
 4fcf0ae:      	pshuflw	$0xd4, %xmm0, %xmm0     # xmm0 = xmm0[0,1,1,3,4,5,6,7]
 4fcf0b3:      	pshufd	$0xd4, %xmm0, %xmm0     # xmm0 = xmm0[0,1,1,3]
 4fcf0b8:      	movdqa	-0x4c47920(%rip), %xmm3 # 0x3877a0 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x480>
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
 4fd1a0e:      	leaq	-0x4c3fd45(%rip), %rdi  # 0x391cd0 <anon.bd10a02e3ec00da32c2ae70ffdaca96c.251.llvm.1341170882515377341>
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
 4fd1cb8:      	movdqa	-0x4c4a530(%rip), %xmm1 # 0x387790 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x470>
 4fd1cc0:      	movdqa	-0x4c4a528(%rip), %xmm3 # 0x3877a0 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x480>
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
 4fd1d42:      	movdqa	-0x4c4a5ba(%rip), %xmm1 # 0x387790 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x470>
 4fd1d4a:      	movdqa	-0x4c4a5b2(%rip), %xmm3 # 0x3877a0 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x480>
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
 4fd1e9e:      	movdqa	-0x4c513e6(%rip), %xmm1 # 0x380ac0 <anon.d42d8d1710be02f83480d8869995a290.139.llvm.4933824046006920891+0x1a0>
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
 4fd8619:      	xorps	-0x4c511a0(%rip), %xmm0 # 0x387480 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x160>
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
 4fd86c4:      	pxor	-0x4c5124c(%rip), %xmm0 # 0x387480 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x160>
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
 4fd86f7:      	xorps	-0x4c5127e(%rip), %xmm0 # 0x387480 <anon.c9df34cb1bba17f6dff0f5d80aaeed8e.52.llvm.7978163781625274282+0x160>
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
