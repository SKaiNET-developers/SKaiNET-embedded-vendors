pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "skainet-embedded-vendors"

include("synaptics-torq")

// To develop against a local SKaiNET checkout instead of the published artifacts,
// uncomment the line below: Gradle substitutes the `sk.ainet.core:*` dependencies
// with the local composite-build projects.
// includeBuild("../SKaiNET")
