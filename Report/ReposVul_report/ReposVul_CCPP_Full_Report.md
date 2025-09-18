# ReposVul C/C++ — Full Dataset Analysis (from real files)

**Dataset dir:** `C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp`


## Split: **train**
- **Records:** 185,791

### Labels
- 0 (non-vuln): 180,259
- 1 (vuln): 5,532 — **2.98%** positives

### Languages (top)
- C: 169,731
- C++: 16,060

### CWE categories (top)
- CWE-119: 21,383
- CWE-20: 18,021
- CWE-125: 16,746
- CWE-787: 15,963
- CWE-476: 14,214
- CWE-416: 13,826
- NVD-CWE-noinfo: 10,974
- CWE-264: 10,365
- CWE-190: 9,356
- CWE-399: 7,963
- CWE-400: 7,374
- CWE-200: 6,572
- CWE-362: 6,398
- CWE-189: 4,175
- CWE-269: 3,191
- CWE-120: 3,158
- CWE-122: 3,134
- CWE-415: 2,971
- CWE-401: 2,779
- CWE-74: 2,629
- CWE-617: 2,530
- CWE-284: 2,118
- CWE-835: 1,908
- CWE-59: 1,820
- CWE-369: 1,665

### Projects (top)
- torvalds/linux: 69,759
- imagemagick/imagemagick: 5,396
- radareorg/radare2: 4,282
- flatpak/flatpak: 3,751
- rdesktop/rdesktop: 3,498
- ffmpeg/ffmpeg: 3,422
- vim/vim: 3,007
- gregkh/linux: 2,484
- libraw/libraw: 2,464
- opensc/opensc: 2,456
- freerdp/freerdp: 2,353
- bminor/binutils-gdb: 2,043
- redis/redis: 2,009
- litespeedtech/lsquic: 1,829
- mozilla/gecko-dev: 1,628
- git/git: 1,602
- postgres/postgres: 1,549
- qemu/qemu: 1,466
- the-tcpdump-group/tcpdump: 1,433
- apache/httpd: 1,243
- uclouvain/openjpeg: 1,209
- mruby/mruby: 1,160
- php/php-src: 1,081
- openvswitch/ovs: 1,023
- krb5/krb5: 1,013

### File names (top)
- arch/x86/kvm/vmx.c: 3,304
- arch/x86/kvm/x86.c: 2,470
- kernel/bpf/verifier.c: 2,241
- internal/dcraw_common.cpp: 2,162
- shlr/java/class.c: 1,990
- common/flatpak-dir.c: 1,828
- kernel/events/core.c: 1,598
- arch/x86/kvm/emulate.c: 1,332
- src/lib/openjp2/j2k.c: 1,044
- virt/kvm/kvm_main.c: 1,029
- net/packet/af_packet.c: 1,028
- fs/namespace.c: 922
- net/socket.c: 868
- rdp.c: 859
- net/netfilter/nf_tables_api.c: 854
- libavformat/movenc.c: 829
- coders/psd.c: 821
- fs/io_uring.c: 804
- Magick++/lib/Image.cpp: 783
- net/xfrm/xfrm_user.c: 780
- fs/namei.c: 717
- net/key/af_key.c: 687
- net/sctp/socket.c: 657
- coders/tiff.c: 646
- kernel/trace/trace.c: 639

### Static analyzer flags (tool → true/false/missing)
- flawfinder: true=20,666, false=165,125, missing=0
- rats: true=9,620, false=176,171, missing=0
- semgrep: true=4,029, false=181,762, missing=0
- cppcheck: true=763, false=185,028, missing=0

### Inter-procedural connectivity — Coverage & Evidence
- `caller` present: 185,791 | non-empty: 23,384 (**12.59%**)
- `callee` present: 185,791 | non-empty: 53,794 (**28.95%**)
- **Both** caller & callee non-empty: 16,206 (**8.72%**)
- `caller_of_change` present: 185,791 | non-empty: 857 (**0.46%**)
- `callee_of_change` present: 185,791 | non-empty: 5,187 (**2.79%**)
- **Both** change-only non-empty: 112 (**0.06%**)

#### Concrete samples (proof)
- **Chain (caller → function_id → callee):** `MagickCore.locale.LocaleCompare` → `01faddbe2711a4156180c4a92837e2f23683cc68_27` → `MagickCore.magick.MagickCoreTerminus`
- Example `caller` for `01faddbe2711a4156180c4a92837e2f23683cc68_27`: {"callers": ["MagickCore.locale.LocaleCompare"]}
- Example `callee` for `01faddbe2711a4156180c4a92837e2f23683cc68_27`: {"callees": ["MagickCore.magick.MagickCoreTerminus"]}
- Example `caller_of_change` for `a63893791280d441c713293491da97c79c0950fe_3`: {"callers_of_change": ["ecc-mod-arith.ecc_mod_sqr", "ecc-mod-arith.ecc_mod_mul", "mini-gmp.mpn_sub_n", "cnd-copy.cnd_copy"]}
- Example `callee_of_change` for `296debd213bd6dce7647cedd34eb64e5b94cdc92_8`: {"callees_of_change": ["libavcodec.dnxhddec.dnxhd_decode_header"]}

#### Edge CSV exports
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\train_c_cpp_repository2__caller_edges.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\train_c_cpp_repository2__callee_edges.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\train_c_cpp_repository2__caller_edges_change.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\train_c_cpp_repository2__callee_edges_change.csv

## Split: **valid**
- **Records:** 23,224

### Labels
- 0 (non-vuln): 22,503
- 1 (vuln): 721 — **3.10%** positives

### Languages (top)
- C: 21,218
- C++: 2,006

### CWE categories (top)
- CWE-119: 2,601
- CWE-20: 2,150
- CWE-125: 2,085
- CWE-787: 2,041
- CWE-476: 1,803
- CWE-416: 1,767
- NVD-CWE-noinfo: 1,368
- CWE-264: 1,314
- CWE-190: 1,177
- CWE-399: 1,057
- CWE-400: 894
- CWE-200: 822
- CWE-362: 772
- CWE-189: 539
- CWE-269: 441
- CWE-120: 394
- CWE-122: 391
- CWE-415: 380
- CWE-401: 367
- CWE-74: 333
- CWE-617: 311
- CWE-284: 292
- CWE-59: 241
- CWE-404: 227
- CWE-835: 223

### Projects (top)
- torvalds/linux: 8,628
- imagemagick/imagemagick: 665
- radareorg/radare2: 522
- rdesktop/rdesktop: 476
- flatpak/flatpak: 439
- vim/vim: 394
- ffmpeg/ffmpeg: 390
- opensc/opensc: 329
- gregkh/linux: 305
- libraw/libraw: 293
- bminor/binutils-gdb: 277
- redis/redis: 276
- freerdp/freerdp: 266
- litespeedtech/lsquic: 225
- mozilla/gecko-dev: 219
- git/git: 218
- postgres/postgres: 205
- qemu/qemu: 181
- mruby/mruby: 164
- the-tcpdump-group/tcpdump: 160
- apache/httpd: 159
- libvirt/libvirt: 150
- uclouvain/openjpeg: 140
- php/php-src: 136
- openvswitch/ovs: 131

### File names (top)
- arch/x86/kvm/vmx.c: 410
- arch/x86/kvm/x86.c: 335
- kernel/bpf/verifier.c: 268
- internal/dcraw_common.cpp: 253
- shlr/java/class.c: 231
- common/flatpak-dir.c: 198
- kernel/events/core.c: 197
- arch/x86/kvm/emulate.c: 163
- rdp.c: 130
- net/packet/af_packet.c: 128
- virt/kvm/kvm_main.c: 122
- net/xfrm/xfrm_user.c: 119
- src/lib/openjp2/j2k.c: 115
- net/netfilter/nf_tables_api.c: 113
- fs/namespace.c: 111
- libavformat/movenc.c: 98
- coders/psd.c: 94
- fs/io_uring.c: 94
- coders/tiff.c: 91
- Magick++/lib/Image.cpp: 90
- memcached.c: 90
- net/socket.c: 88
- net/key/af_key.c: 87
- kernel/trace/trace.c: 84
- net/core/dev.c: 83

### Static analyzer flags (tool → true/false/missing)
- flawfinder: true=2,579, false=20,645, missing=0
- rats: true=1,188, false=22,036, missing=0
- semgrep: true=482, false=22,742, missing=0
- cppcheck: true=94, false=23,130, missing=0

### Inter-procedural connectivity — Coverage & Evidence
- `caller` present: 23,224 | non-empty: 3,042 (**13.10%**)
- `callee` present: 23,224 | non-empty: 6,795 (**29.26%**)
- **Both** caller & callee non-empty: 2,140 (**9.21%**)
- `caller_of_change` present: 23,224 | non-empty: 127 (**0.55%**)
- `callee_of_change` present: 23,224 | non-empty: 636 (**2.74%**)
- **Both** change-only non-empty: 17 (**0.07%**)

#### Concrete samples (proof)
- **Chain (caller → function_id → callee):** `common.flatpak-utils.flatpak_xml_free` → `6cac99dafe6003c8a4bd5666341c217876536869_194` → `common.flatpak-utils.validate_component`
- Example `caller` for `6cac99dafe6003c8a4bd5666341c217876536869_194`: {"callers": ["common.flatpak-utils.flatpak_xml_free"]}
- Example `callee` for `6cac99dafe6003c8a4bd5666341c217876536869_194`: {"callees": ["common.flatpak-utils.validate_component", "common.flatpak-utils.flatpak_appstream_xml_filter", "common.flatpak-utils.flatpak_xml_free"]}
- Example `caller_of_change` for `ffb35baac6981f9e8914f8f3bffd37f284b85970_49`: {"callers_of_change": ["src.lib.krb5.krb.princ_comp.krb5_principal_compare"]}
- Example `callee_of_change` for `336a98feb0d56b9ac54e12736b18785c27f75090_171`: {"callees_of_change": ["lib.nghttp2_session.nghttp2_session_mem_recv"]}

#### Edge CSV exports
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\valid_c_cpp_repository2__caller_edges.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\valid_c_cpp_repository2__callee_edges.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\valid_c_cpp_repository2__caller_edges_change.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\valid_c_cpp_repository2__callee_edges_change.csv

## Split: **test**
- **Records:** 23,224

### Labels
- 0 (non-vuln): 22,554
- 1 (vuln): 670 — **2.88%** positives

### Languages (top)
- C: 21,170
- C++: 2,054

### CWE categories (top)
- CWE-119: 2,717
- CWE-20: 2,299
- CWE-787: 2,052
- CWE-125: 2,014
- CWE-476: 1,720
- CWE-416: 1,699
- NVD-CWE-noinfo: 1,378
- CWE-264: 1,263
- CWE-190: 1,229
- CWE-399: 995
- CWE-400: 960
- CWE-200: 829
- CWE-362: 809
- CWE-189: 521
- CWE-120: 429
- CWE-122: 398
- CWE-269: 393
- CWE-415: 372
- CWE-401: 371
- CWE-74: 330
- CWE-617: 310
- CWE-284: 277
- CWE-59: 242
- CWE-369: 219
- CWE-404: 214

### Projects (top)
- torvalds/linux: 8,587
- imagemagick/imagemagick: 668
- radareorg/radare2: 530
- flatpak/flatpak: 457
- rdesktop/rdesktop: 434
- ffmpeg/ffmpeg: 412
- vim/vim: 391
- opensc/opensc: 311
- libraw/libraw: 302
- gregkh/linux: 294
- redis/redis: 287
- freerdp/freerdp: 276
- bminor/binutils-gdb: 273
- litespeedtech/lsquic: 215
- git/git: 212
- mozilla/gecko-dev: 211
- postgres/postgres: 199
- qemu/qemu: 191
- the-tcpdump-group/tcpdump: 178
- php/php-src: 170
- apache/httpd: 156
- mruby/mruby: 156
- uclouvain/openjpeg: 144
- openvswitch/ovs: 139
- libvirt/libvirt: 128

### File names (top)
- arch/x86/kvm/vmx.c: 411
- arch/x86/kvm/x86.c: 351
- kernel/bpf/verifier.c: 291
- internal/dcraw_common.cpp: 254
- shlr/java/class.c: 239
- common/flatpak-dir.c: 209
- kernel/events/core.c: 206
- arch/x86/kvm/emulate.c: 170
- net/packet/af_packet.c: 125
- virt/kvm/kvm_main.c: 122
- src/lib/openjp2/j2k.c: 119
- rdp.c: 113
- net/netfilter/nf_tables_api.c: 111
- fs/namespace.c: 106
- coders/psd.c: 104
- Magick++/lib/Image.cpp: 100
- libavformat/movenc.c: 93
- fs/io_uring.c: 91
- coders/tiff.c: 91
- net/socket.c: 90
- net/xfrm/xfrm_user.c: 90
- net/key/af_key.c: 86
- net/sctp/socket.c: 83
- fs/namei.c: 82
- net/core/dev.c: 81

### Static analyzer flags (tool → true/false/missing)
- flawfinder: true=2,537, false=20,687, missing=0
- rats: true=1,189, false=22,035, missing=0
- semgrep: true=502, false=22,722, missing=0
- cppcheck: true=99, false=23,125, missing=0

### Inter-procedural connectivity — Coverage & Evidence
- `caller` present: 23,224 | non-empty: 3,013 (**12.97%**)
- `callee` present: 23,224 | non-empty: 6,775 (**29.17%**)
- **Both** caller & callee non-empty: 2,096 (**9.03%**)
- `caller_of_change` present: 23,224 | non-empty: 110 (**0.47%**)
- `callee_of_change` present: 23,224 | non-empty: 674 (**2.90%**)
- **Both** change-only non-empty: 20 (**0.09%**)

#### Concrete samples (proof)
- **Chain (caller → function_id → callee):** `src.strings.vim_snprintf` → `37f47958b8a2a44abc60614271d9537e7f14e51a_12` → `src.diff.diff_file`
- Example `caller` for `37f47958b8a2a44abc60614271d9537e7f14e51a_12`: {"callers": ["src.strings.vim_snprintf"]}
- Example `callee` for `d561990a358899178115e156871cc054a1c55ffe_14`: {"callees": ["modules.pico_ipfilter.filter_match_packet"]}
- Example `caller_of_change` for `48361c411e50826cb602c7aab773a8a20e1da6bc_11`: {"callers_of_change": ["winpr.libwinpr.utils.stream.Stream_Free"]}
- Example `callee_of_change` for `97b153237c256c586e528eac7fc2f51aedb2b2fc_0`: {"callees_of_change": ["devicemodel.hw.pci.core.pci_emul_capwrite", "devicemodel.hw.pci.core.pci_emul_cmdsts_write", "devicemodel.hw.pci.core.pci_cfgrw"]}

#### Edge CSV exports
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\test_c_cpp_repository2__caller_edges.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\test_c_cpp_repository2__callee_edges.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\test_c_cpp_repository2__caller_edges_change.csv
- C:\Users\USER\Desktop\Thesis Research\Implementation\Thesis-causal-vul\data\dataset\ReposVul_c_cpp\_edges\test_c_cpp_repository2__callee_edges_change.csv