fn main() {
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .file("native/macos_tap.m")
            .flag("-fobjc-arc")
            .flag("-Wno-deprecated-declarations")
            .compile("macos_tap");

        println!("cargo:rustc-link-lib=framework=CoreAudio");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rerun-if-changed=native/macos_tap.m");
    }
}
