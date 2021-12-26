import 'dart:async';
import 'dart:isolate';
import 'package:flutter/services.dart';
import 'data/methods.dart';
import 'modal/ProcessImage.dart';

class OpencvAwesome {
  static const MethodChannel _channel =
      const MethodChannel('opencv_awesome');
  static Future<String> get platformVersion async {
    final String version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }

 static Future<void> generatePanorama(String path, List<String> angleTilt, List<String> angleLean) async {
    final port = ReceivePort();
    final args = ProcessImageArguments(path, angleTilt.toString(), angleLean.toString());
    Isolate.spawn<ProcessImageArguments>(
        Methods.stitch,
        args,
        onError: port.sendPort,
        onExit: port.sendPort
    );
  }
}

