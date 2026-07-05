plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// group / version are inherited from the root project (gradle.properties: GROUP / VERSION_NAME).

kotlin {
    jvm()

    sourceSets {
        val jvmMain by getting {
            dependencies {
                // Published SKaiNET core (Maven Central) — the TargetOptimizer seam and
                // the DAG/lang types these passes plug into. Versions from libs.versions.toml.
                implementation(libs.skainet.compile.opt)
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.lang.core)
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
    }
}
