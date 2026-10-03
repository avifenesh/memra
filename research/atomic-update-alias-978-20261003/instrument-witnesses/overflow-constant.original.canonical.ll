define void @_RNvMs0_NtNtCskjzQ6PhiQTT_8memra_kv6tiered12materializerNtB5_16QwenMaterializer3new(ptr dead_on_unwind noalias noundef writable writeonly sret([432 x i8]) align 8 captures(none) dereferenceable(432) %v8, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(128) %v9, ptr dead_on_return noalias noundef readonly align 4 captures(none) dereferenceable(324) %v10, ptr dead_on_return noalias noundef readonly align 8 captures(none) dereferenceable(32) %v11) unnamed_addr ATTR{ nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" } personality ptr @rust_eh_personality {
v0:
  %v12 = alloca [8 x i8], align 8
  %v13 = load atomic i64, ptr @_RNvNtNtCskjzQ6PhiQTT_8memra_kv6tiered12materializer17NEXT_MATERIALIZER monotonic, align 8
  br label %v1
v1:
  %v14 = phi i64 [ %v13, %v0 ], [ %v15, %v2 ]
  %v16 = icmp eq i64 %v14, -1
  br i1 %v16, label %v3, label %v2
v2:
  %v17 = add nuw i64 %v14, 1
  %v18 = cmpxchg weak ptr @_RNvNtNtCskjzQ6PhiQTT_8memra_kv6tiered12materializer17NEXT_MATERIALIZER, i64 %v14, i64 %v17 monotonic monotonic, align 8
  %v19 = extractvalue { i64, i1 } %v18, 1
  %v15 = extractvalue { i64, i1 } %v18, 0
  br i1 %v19, label %v4, label %v1
v3:
  call void @llvm.lifetime.start.p0(ptr nonnull %v12), !noalias META0(!{META1(distinct !{META1, META2(distinct !{META2, !"_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultyyE6expectCskjzQ6PhiQTT_8memra_kv"}), !"_RNvMNtCs4NRVxsYgnAr_4core6resultINtB2_6ResultyyE6expectCskjzQ6PhiQTT_8memra_kv: argument 0"})})
  store i64 -1, ptr %v12, align 8, !noalias META0
  call void @_RNvNtCs4NRVxsYgnAr_4core6result13unwrap_failed(ptr noalias noundef nonnull readonly captures(address, read_provenance) @alloc_a0f07ab120f6af5d94998a62eeb2628a, i64 noundef 25, ptr noundef nonnull %v12, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(32) @vtable.g, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_e0b23e4045dcc137494492333a59ecba) ATTR{ noreturn }
  unreachable
v4:
  %v20 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @_RNvNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBa_11RandomState3new4KEYS0s_023___RUST_STD_INTERNAL_VAL)
  %v21 = getelementptr inbounds nuw i8, ptr %v20, i64 16
  %v22 = load i8, ptr %v21, align 8, !range META3(!{i8 0, i8 2}), !noalias META4(!{META5(distinct !{META5, META6(distinct !{META6, !"_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE11get_or_initNvNvNvMNtNtBe_4hash6randomNtB2d_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv"}), !"_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE11get_or_initNvNvNvMNtNtBe_4hash6randomNtB2d_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv: %i"}), META7(distinct !{META7, META8(distinct !{META8, !"_RNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtB8_11RandomState3new4KEYS0s_0CskjzQ6PhiQTT_8memra_kv"}), !"_RNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtB8_11RandomState3new4KEYS0s_0CskjzQ6PhiQTT_8memra_kv: %__rust_std_internal_init"}), META9(distinct !{META9, META10(distinct !{META10, !"_RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv"}), !"_RNvYNCNKNvNvMNtNtCs2AWtUsOyxgP_3std4hash6randomNtBb_11RandomState3new4KEYS0s_0INtNtNtCs4NRVxsYgnAr_4core3ops8function6FnOnceTINtNtB1l_6option6OptionQIB20_INtNtB1l_4cell4CellTyyEEEEEE9call_onceCskjzQ6PhiQTT_8memra_kv: argument 0"}), META11(distinct !{META11, META12(distinct !{META12, !"_RINvMs2_NtNtCs2AWtUsOyxgP_3std6thread5localINtB6_8LocalKeyINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEE8try_withNCNvMNtNtBa_4hash6randomNtB1M_11RandomState3new0B25_ECskjzQ6PhiQTT_8memra_kv"}), !"_RINvMs2_NtNtCs2AWtUsOyxgP_3std6thread5localINtB6_8LocalKeyINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEE8try_withNCNvMNtNtBa_4hash6randomNtB1M_11RandomState3new0B25_ECskjzQ6PhiQTT_8memra_kv: %_0"})}), !noundef META13(!{})
  %v23 = trunc nuw i8 %v22 to i1
  br i1 %v23, label %v5, label %v6, !prof META14(!{!"branch_weights", !"expected", i32 2000, i32 1})
v5:
  %v24 = load i64, ptr %v20, align 8, !noalias META15(!{META11})
  %v25 = getelementptr inbounds nuw i8, ptr %v20, i64 8
  %v26 = load i64, ptr %v25, align 8, !noalias META15
  br label %v7
v6:
  %v27 = tail call { i64, i64 } @_RNvNtNtNtCs2AWtUsOyxgP_3std3sys6random5linux19hashmap_random_keys(), !noalias META16(!{META17(distinct !{META17, META18(distinct !{META18, !"_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv"}), !"_RINvMs0_NtNtNtNtCs2AWtUsOyxgP_3std3sys12thread_local6native4lazyINtB6_7StorageINtNtCs4NRVxsYgnAr_4core4cell4CellTyyEEzE16get_or_init_slowNvNvNvMNtNtBe_4hash6randomNtB2i_11RandomState3new4KEYS27___rust_std_internal_init_fnECskjzQ6PhiQTT_8memra_kv: argument 0"}), META11})
  %v28 = extractvalue { i64, i64 } %v27, 0
  %v29 = extractvalue { i64, i64 } %v27, 1
  %v30 = getelementptr inbounds nuw i8, ptr %v20, i64 8
  store i64 %v29, ptr %v30, align 8, !noalias META16
  store i8 1, ptr %v21, align 8, !noalias META16
  br label %v7
v7:
  %v31 = phi i64 [ %v26, %v5 ], [ %v29, %v6 ]
  %v32 = phi i64 [ %v24, %v5 ], [ %v28, %v6 ]
  %v33 = add i64 %v32, 1
  store i64 %v33, ptr %v20, align 8, !noalias META15
  %v34 = getelementptr inbounds nuw i8, ptr %v8, i64 56
  store i64 %v15, ptr %v34, align 8
  store ptr %v9, ptr %v8, align 8
  %v35 = getelementptr inbounds nuw i8, ptr %v8, i64 104
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(324) %v35, ptr noundef nonnull align 4 dereferenceable(324) %v10, i64 324, i1 false)
  %v36 = getelementptr inbounds nuw i8, ptr %v8, i64 64
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %v36, ptr noundef nonnull align 8 dereferenceable(32) %v11, i64 32, i1 false)
  %v37 = getelementptr inbounds nuw i8, ptr %v8, i64 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %v37, ptr noundef nonnull align 8 dereferenceable(32) @anon.46f30e44168cea28848fd5983f4a9cf9.0, i64 32, i1 false)
  %v38 = getelementptr inbounds nuw i8, ptr %v8, i64 40
  store i64 %v32, ptr %v38, align 8
  %v39 = getelementptr inbounds nuw i8, ptr %v8, i64 48
  store i64 %v31, ptr %v39, align 8
  %v40 = getelementptr inbounds nuw i8, ptr %v8, i64 96
  store i64 0, ptr %v40, align 8
  ret void
}
