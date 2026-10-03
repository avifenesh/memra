define void @_RNvMs0_NtNtCskjzQ6PhiQTT_8memra_kv6tiered12materializerNtB5_16QwenMaterializer3new(ptr dead_on_unwind noalias noundef writable writeonly sret([432 x i8]) align 8 captures(none) dereferenceable(432) %_0, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(128) %owner, ptr dead_on_return noalias noundef readonly align 4 captures(none) dereferenceable(324) %program, ptr dead_on_return noalias noundef readonly align 8 captures(none) dereferenceable(32) %geometry) unnamed_addr #3 personality ptr @rust_eh_personality {
start:
  %e.i = alloca [8 x i8], align 8
  %0 = load atomic i64, ptr @_RNvNtNtCskjzQ6PhiQTT_8memra_kv6tiered12materializer17NEXT_MATERIALIZER monotonic, align 8
  br label %bb1.i

bb1.i:                                            ; preds = %bb3.i, %start
  %prev.sroa.0.0.i = phi i64 [ %0, %start ], [ %3, %bb3.i ]
  %_4.1.i.not.i = icmp eq i64 %prev.sroa.0.0.i, -1
  br i1 %_4.1.i.not.i, label %bb2.i, label %bb3.i

bb3.i:                                            ; preds = %bb1.i
  %_4.0.i.i = add nuw i64 %prev.sroa.0.0.i, 1
  %1 = cmpxchg weak ptr @_RNvNtNtCskjzQ6PhiQTT_8memra_kv6tiered12materializer17NEXT_MATERIALIZER, i64 %prev.sroa.0.0.i, i64 %_4.0.i.i monotonic monotonic, align 8
  %2 = extractvalue { i64, i1 } %1, 1
  %3 = extractvalue { i64, i1 } %1, 0
  br i1 %2, label %_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultyyE6expectCskjzQ6PhiQTT_8memra_kv.exit, label %bb1.i

bb2.i:                                            ; preds = %bb1.i
  call void @llvm.lifetime.start.p0(ptr nonnull %e.i), !noalias !2448
  store i64 -1, ptr %e.i, align 8, !noalias !2448
; call core::result::unwrap_failed
  call void @_RNvNtCs4NRVxsYgnAr_4core6result13unwrap_failed(ptr noalias noundef nonnull readonly captures(address, read_provenance) @alloc_a0f07ab120f6af5d94998a62eeb2628a, i64 noundef 25, ptr noundef nonnull %e.i, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(32) @vtable.g, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_e0b23e4045dcc137494492333a59ecba) #55
  unreachable

_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultyyE6expectCskjzQ6PhiQTT_8memra_kv.exit: ; preds = %bb3.i
  %_3.i.i.i.i = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @_RNvNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBa_11RandomState3new4KEYS0s_023___RUST_STD_INTERNAL_VAL)
  %_12.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %_3.i.i.i.i, i64 16
  %4 = load i8, ptr %_12.i.i.i.i.i, align 8, !range !752, !noalias !2451, !noundef !17
  %_4.i.i.i.i.i = trunc nuw i8 %4 to i1
  br i1 %_4.i.i.i.i.i, label %start._RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv.exit_crit_edge.i.i, label %_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv.exit.i.i, !prof !37

start._RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv.exit_crit_edge.i.i: ; preds = %_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultyyE6expectCskjzQ6PhiQTT_8memra_kv.exit
  %_9.i.pre.i.i = load i64, ptr %_3.i.i.i.i, align 8, !noalias !2460
  %.phi.trans.insert.i.i = getelementptr inbounds nuw i8, ptr %_3.i.i.i.i, i64 8
  %_10.i.pre.i.i = load i64, ptr %.phi.trans.insert.i.i, align 8, !noalias !2460
  br label %_RINvMs2_NtNtCs2AWtUsOyxgP_3std6thread5localINtB6_8LocalKeyINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEE4withNCNvMNtNtBa_4hash6randomNtB1I_11RandomState3new0B21_ECskjzQ6PhiQTT_8memra_kv.exit

_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv.exit.i.i: ; preds = %_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultyyE6expectCskjzQ6PhiQTT_8memra_kv.exit
; call std::sys::random::linux::hashmap_random_keys
  %5 = tail call { i64, i64 } @_RNvNtNtNtCs2AWtUsOyxgP_3std3sys6random5linux19hashmap_random_keys(), !noalias !2461
  %6 = extractvalue { i64, i64 } %5, 0
  %7 = extractvalue { i64, i64 } %5, 1
  %8 = getelementptr inbounds nuw i8, ptr %_3.i.i.i.i, i64 8
  store i64 %7, ptr %8, align 8, !noalias !2461
  store i8 1, ptr %_12.i.i.i.i.i, align 8, !noalias !2461
  br label %_RINvMs2_NtNtCs2AWtUsOyxgP_3std6thread5localINtB6_8LocalKeyINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEE4withNCNvMNtNtBa_4hash6randomNtB1I_11RandomState3new0B21_ECskjzQ6PhiQTT_8memra_kv.exit

_RINvMs2_NtNtCs2AWtUsOyxgP_3std6thread5localINtB6_8LocalKeyINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEE4withNCNvMNtNtBa_4hash6randomNtB1I_11RandomState3new0B21_ECskjzQ6PhiQTT_8memra_kv.exit: ; preds = %start._RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv.exit_crit_edge.i.i, %_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv.exit.i.i
  %_7.1.pre-phi = phi i64 [ %_10.i.pre.i.i, %start._RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv.exit_crit_edge.i.i ], [ %7, %_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv.exit.i.i ]
  %_9.i.i.i = phi i64 [ %_9.i.pre.i.i, %start._RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv.exit_crit_edge.i.i ], [ %6, %_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv.exit.i.i ]
  %_4.i.i.i = add i64 %_9.i.i.i, 1
  store i64 %_4.i.i.i, ptr %_3.i.i.i.i, align 8, !noalias !2460
  %9 = getelementptr inbounds nuw i8, ptr %_0, i64 56
  store i64 %3, ptr %9, align 8
  store ptr %owner, ptr %_0, align 8
  %10 = getelementptr inbounds nuw i8, ptr %_0, i64 104
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(324) %10, ptr noundef nonnull align 4 dereferenceable(324) %program, i64 324, i1 false)
  %11 = getelementptr inbounds nuw i8, ptr %_0, i64 64
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %geometry, i64 32, i1 false)
  %12 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) @anon.46f30e44168cea28848fd5983f4a9cf9.0, i64 32, i1 false)
  %_6.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 40
  store i64 %_9.i.i.i, ptr %_6.sroa.4.0..sroa_idx, align 8
  %_6.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 48
  store i64 %_7.1.pre-phi, ptr %_6.sroa.5.0..sroa_idx, align 8
  %13 = getelementptr inbounds nuw i8, ptr %_0, i64 96
  store i64 0, ptr %13, align 8
  ret void
}
