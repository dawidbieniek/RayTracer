#line 1 "D:\\Projekty\\RayTracer\\main.cu"
static const struct __si_class_type_info _ZTISt20bad_array_new_length;
static const struct __si_class_type_info _ZTISt12system_error;
static const struct __si_class_type_info _ZTISt8bad_cast;
static const struct __si_class_type_info _ZTINSt8ios_base7failureE;
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22000.0\\ucrt\\corecrt_stdio_config.h"
 /* COMDAT group: _ZZ28__local_stdio_printf_optionsE15_OptionsStorage */ unsigned __int64 _ZZ28__local_stdio_printf_optionsE15_OptionsStorage;
#line 163 "D:\\Projekty\\RayTracer\\main.cu"
struct vec3 *fb = 0;

extern int lastFrameTime;
int deltaTimes[60];
extern int deltaTimesIndex;
extern void *__dso_handle;
#line 669 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\system_error"
 /* COMDAT group: _ZZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static */ struct _ZSt27_Constexpr_immortalize_implISt25_Iostream_error_category2E _ZZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static = {{{{0}}}};
 /* COMDAT group: _ZZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static */ unsigned __int64 _ZGVZSt25_Immortalize_memcpy_imageISt25_Iostream_error_category2ERKT_vE7_Static;
extern struct __C7 *__curr_eh_stack_entry;
extern unsigned short __eh_curr_region;
extern int __catch_clause_number;
static const struct __class_type_info _ZTISt9exception;
extern  /* COMDAT group: _ZTSSt9exception */ const char _ZTSSt9exception[13];
static const struct __si_class_type_info _ZTISt9bad_alloc;
extern  /* COMDAT group: _ZTSSt9bad_alloc */ const char _ZTSSt9bad_alloc[13];
extern  /* COMDAT group: _ZTSSt20bad_array_new_length */ const char _ZTSSt20bad_array_new_length[25];
static const struct __si_class_type_info _ZTISt13runtime_error;
extern  /* COMDAT group: _ZTSSt13runtime_error */ const char _ZTSSt13runtime_error[18];
static const struct __si_class_type_info _ZTISt13_System_error;
extern  /* COMDAT group: _ZTSSt13_System_error */ const char _ZTSSt13_System_error[18];
extern  /* COMDAT group: _ZTSSt12system_error */ const char _ZTSSt12system_error[17];
extern  /* COMDAT group: _ZTSSt8bad_cast */ const char _ZTSSt8bad_cast[12];
extern  /* COMDAT group: _ZTSNSt8ios_base7failureE */ const char _ZTSNSt8ios_base7failureE[22];
#line 115 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\xlocale"
extern __declspec( dllimport ) int _ZNSt6locale2id7_Id_cntE;
#line 102 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\xlocnum"
extern __declspec( dllimport ) struct _ZNSt6locale2idE _ZNSt8numpunctIcE2idE;
#line 1174 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\xlocnum"
extern __declspec( dllimport ) struct _ZNSt6locale2idE _ZNSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE2idE;
#line 102 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\xlocnum"
extern __declspec( dllimport ) struct _ZNSt6locale2idE _ZNSt8numpunctIwE2idE;
#line 419 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\xlocale"
extern  /* COMDAT group: _ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE */ const struct _ZNSt6locale5facetE *_ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE;
#line 42 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\vcruntime_new.h"
extern const struct _ZSt9nothrow_t _ZSt7nothrow;
#line 41 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\iostream"
extern __declspec( dllimport ) _ZSt7ostream _ZSt4cerr;
static const struct __si_class_type_info _ZTISt20bad_array_new_length = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt20bad_array_new_length}},((const struct __class_type_info *)(&_ZTISt9bad_alloc.base))};
static const struct __si_class_type_info _ZTISt12system_error = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt12system_error}},((const struct __class_type_info *)(&_ZTISt13_System_error.base))};
static const struct __si_class_type_info _ZTISt8bad_cast = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt8bad_cast}},(&_ZTISt9exception)};
static const struct __si_class_type_info _ZTINSt8ios_base7failureE = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSNSt8ios_base7failureE}},((const struct __class_type_info *)(&_ZTISt12system_error.base))};
#line 165 "D:\\Projekty\\RayTracer\\main.cu"
int lastFrameTime = 0;

int deltaTimesIndex = 0;
static const struct __class_type_info _ZTISt9exception = {{(_ZTVN10__cxxabiv117__class_type_infoE + 2),_ZTSSt9exception}};
 /* COMDAT group: _ZTSSt9exception */ const char _ZTSSt9exception[13] = "St9exception";
static const struct __si_class_type_info _ZTISt9bad_alloc = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt9bad_alloc}},(&_ZTISt9exception)};
 /* COMDAT group: _ZTSSt9bad_alloc */ const char _ZTSSt9bad_alloc[13] = "St9bad_alloc";
 /* COMDAT group: _ZTSSt20bad_array_new_length */ const char _ZTSSt20bad_array_new_length[25] = "St20bad_array_new_length";
static const struct __si_class_type_info _ZTISt13runtime_error = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt13runtime_error}},(&_ZTISt9exception)};
 /* COMDAT group: _ZTSSt13runtime_error */ const char _ZTSSt13runtime_error[18] = "St13runtime_error";
static const struct __si_class_type_info _ZTISt13_System_error = {{{(_ZTVN10__cxxabiv120__si_class_type_infoE + 2),_ZTSSt13_System_error}},((const struct __class_type_info *)(&_ZTISt13runtime_error.base))};
 /* COMDAT group: _ZTSSt13_System_error */ const char _ZTSSt13_System_error[18] = "St13_System_error";
 /* COMDAT group: _ZTSSt12system_error */ const char _ZTSSt12system_error[17] = "St12system_error";
 /* COMDAT group: _ZTSSt8bad_cast */ const char _ZTSSt8bad_cast[12] = "St8bad_cast";
 /* COMDAT group: _ZTSNSt8ios_base7failureE */ const char _ZTSNSt8ios_base7failureE[22] = "NSt8ios_base7failureE";
#line 419 "E:\\VisualStudio\\2022\\Preview\\VC\\Tools\\MSVC\\14.37.32705\\include\\xlocale"
 /* COMDAT group: _ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE */ const struct _ZNSt6locale5facetE *_ZNSt9_FacetptrISt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEEE6_PsaveE = ((const struct _ZNSt6locale5facetE *)0i64);
