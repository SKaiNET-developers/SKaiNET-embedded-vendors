plugins {
    alias(libs.plugins.kotlinMultiplatform) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
}

allprojects {
    group = providers.gradleProperty("GROUP").getOrElse("sk.ainet.vendors")
    version = providers.gradleProperty("VERSION_NAME").getOrElse("unspecified")
}

// Require JDK 21+ (produces Java 21 bytecode via --release / jvmTarget), mirroring SKaiNET core.
subprojects {
    require(JavaVersion.current() >= JavaVersion.VERSION_21) {
        "This project requires JDK 21+, but found ${JavaVersion.current()}"
    }

    plugins.withId("org.jetbrains.kotlin.multiplatform") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension::class.java)?.apply {
            targets.withType(org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget::class.java) {
                compilerOptions {
                    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
                }
            }
        }
    }

    afterEvaluate {
        tasks.withType<JavaCompile>().configureEach {
            options.release.set(21)
        }
    }
}
