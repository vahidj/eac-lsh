package org.soic.eaclsh

import com.typesafe.config.ConfigFactory

/**
  * Created by vjalali on 3/13/16.
  */
object EACLshConfig {
  val config = ConfigFactory.load("application.conf")
  lazy val BASEPATH = config.getString("eac.base.path")
}
