plugins {
    kotlin("multiplatform") version "2.3.21"
}

group = "sk.ainet.vendors"
version = "0.1.0"

repositories {
    mavenCentral()
    google()
}

kotlin {
    jvm()
    sourceSets {
        val jvmMain by getting {
            dependencies {
                // Substituted by the local SKaiNET projects via the composite build
                // (settings.gradle.kts). Version is a placeholder; the substitution matches
                // on group:name and ignores it.
                implementation("sk.ainet.core:skainet-compile-opt:0.34.0")
                implementation("sk.ainet.core:skainet-compile-dag:0.34.0")
                implementation("sk.ainet.core:skainet-lang-core:0.34.0")
            }
        }
    }
}
